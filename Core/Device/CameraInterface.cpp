#include "CameraInterface.h"

// camera output config realted static objects
uint staticWidth          = 0;
uint staticHeight         = 0;
uint staticBytesPerPixel  = 0;

#if defined(__APPLE__) && defined(USE_METAL_COMPUTE)

#import <AVKit/AVKit.h>

typedef NS_ENUM(NSInteger, AVCamSetupResult) {
    AVCamSetupResultSuccess,
    AVCamSetupResultCameraNotAuthorized,
    AVCamSetupResultSessionConfigurationFailed
};

// native camera realted static objects
AVCamSetupResult staticCameraSetupResult = AVCamSetupResultCameraNotAuthorized;
AVCaptureSession            *staticSession            = NULL;
AVCaptureVideoPreviewLayer  *staticVideoPreviewLayer  = NULL;
AVCaptureVideoDataOutput    *staticVideoDataOutput    = NULL;
AVCaptureDeviceInput        *staticDeviceInput        = NULL;
CVMetalTextureCacheRef      staticMetalTextureCache   = NULL;
dispatch_queue_t            staticSessionQueue;
ComputeInterface*           staticCompute = NULL;

#define MAX_INFLIGHT_COMMAND_BUFFERS 3
ComputeTextureIdentifier staticMetalTextures[MAX_INFLIGHT_COMMAND_BUFFERS];
ComputeTexture staticMetalTexture;
uint staticMetalTextureIndex;

CameraInterface::CameraInterface(ComputeInterface* compute)
{
  staticCompute = compute;
}

CameraInterface::~CameraInterface()
{
  if (staticSession && [staticSession isRunning])
  {
    [staticSession stopRunning];
  }
}

/*!@class Interface to process camera output.*/
@interface SampleBufferDeligate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>

- (void)captureOutput:(AVCaptureOutput*)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection*)connection;

@end

@implementation SampleBufferDeligate

#pragma mark -
#pragma mark AVCaptureVideoDataOutputSampleBufferDelegate

- (void)captureOutput:(AVCaptureOutput*)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection*)connection
{
  staticMetalTextureIndex = (staticMetalTextureIndex + 1) % MAX_INFLIGHT_COMMAND_BUFFERS;

  CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);

  CVPixelBufferLockBaseAddress(pixelBuffer, 0);
  staticWidth = (uint)CVPixelBufferGetWidth(pixelBuffer);
  staticHeight = (uint)CVPixelBufferGetHeight(pixelBuffer);

  CVMetalTextureRef textureRef;

  CVReturn error = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, staticMetalTextureCache, pixelBuffer, NULL, MTLPixelFormatRGBA8Uint, staticWidth, staticHeight, 0, &textureRef);

  if (error)
  {
    logComputeError("Could not create texture from image");
  }

  staticMetalTextures[staticMetalTextureIndex] = CVMetalTextureGetTexture(textureRef);
  if (!staticMetalTextures[staticMetalTextureIndex])
  {
    logComputeError("Could not get texture from texture ref");
  }

  CVBufferRelease(textureRef);
  CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
}

@end

#define CAMERA_SETUP_FAILED staticCameraSetupResult = AVCamSetupResultSessionConfigurationFailed;\
  [staticSession commitConfiguration];

// Call this on the session queue.
void CameraInterface::startSession()
{
  staticSessionQueue = dispatch_queue_create("Camera Session Queue", DISPATCH_QUEUE_SERIAL);
  staticCameraSetupResult = AVCamSetupResultSuccess;

  // Check video authorization status
  // Video access is required and audio access is optional
  switch ([AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo])
  {
    case AVAuthorizationStatusAuthorized:
    {
      // The user has previously granted access to the camera
      break;
    }
    case AVAuthorizationStatusNotDetermined:
    {
      // The user has not yet been presented with the option to grant video access
      // We suspend the session queue to delay session setup until the access request has completed
      // Note that audio access will be implicitly requested when we create an AVCaptureDeviceInput for audio during session setup
      dispatch_suspend(staticSessionQueue);
      [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^(BOOL granted) {
        if (!granted)
        {
          staticCameraSetupResult = AVCamSetupResultCameraNotAuthorized;
        }
        dispatch_resume(staticSessionQueue);
      }];
      break;
    }
    default:
    {
      // The user has previously denied access.
      staticCameraSetupResult = AVCamSetupResultCameraNotAuthorized;
      return;
    }
  }

  if (staticCameraSetupResult == AVCamSetupResultCameraNotAuthorized)
  {
    return;
  }

  staticBytesPerPixel = 4;
  staticMetalTextureIndex = 0;

  // Create a AVCaptureSession
  staticSession = [[AVCaptureSession alloc] init];

  // initialize texture cache
  CVMetalTextureCacheFlush(staticMetalTextureCache, 0);
  CVReturn textureCacheError = CVMetalTextureCacheCreate(kCFAllocatorDefault, NULL, staticCompute->getDevice(), NULL, &staticMetalTextureCache);

  if (textureCacheError)
  {
    logComputeError("Could not create a texture cache");
  }

  [staticSession beginConfiguration];
  [staticSession setSessionPreset:AVCaptureSessionPresetMedium];

  // Add video input
  // Choose the back dual camera if available, otherwise default to a wide angle camera
#if TARGET_OS_IPHONE
  AVCaptureDevice* videoDevice = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInDualCamera
                                                                    mediaType:AVMediaTypeVideo
                                                                     position:AVCaptureDevicePositionBack];
#else
  AVCaptureDevice* videoDevice = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInWideAngleCamera
                                                                    mediaType:AVMediaTypeVideo
                                                                     position:AVCaptureDevicePositionBack];
#endif

  // If a rear dual camera is not available, default to the rear wide angle camera.
  if (!videoDevice)
  {
    videoDevice = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInWideAngleCamera
                                                     mediaType:AVMediaTypeVideo
                                                      position:AVCaptureDevicePositionBack];
  }

  // In the event that the rear wide angle camera isn't available, default to the front wide angle camera.
  if (!videoDevice)
  {
    videoDevice = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInWideAngleCamera
                                                     mediaType:AVMediaTypeVideo
                                                      position:AVCaptureDevicePositionFront];
  }

  NSError* error = nil;
  staticDeviceInput = [AVCaptureDeviceInput deviceInputWithDevice:videoDevice error:&error];

  if (!staticDeviceInput)
  {
    CAMERA_SETUP_FAILED
    logComputeError("Could not create video device input: %s", error);
  }

  if ([staticSession canAddInput:staticDeviceInput])
  {
    [staticSession addInput:staticDeviceInput];

    staticVideoPreviewLayer = [[AVCaptureVideoPreviewLayer alloc] initWithSession:staticSession];
    staticVideoPreviewLayer.connection.videoOrientation = AVCaptureVideoOrientationLandscapeRight;
#if TARGET_OS_IPHONE
    [staticVideoPreviewLayer.connection setPreferredVideoStabilizationMode:AVCaptureVideoStabilizationModeOff];
#endif
  }
  else
  {
    CAMERA_SETUP_FAILED
    logComputeError("Could not add video device input to the session");
  }

  // Add audio input
//  AVCaptureDevice* audioDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeAudio];
//  AVCaptureDeviceInput* audioDeviceInput = [AVCaptureDeviceInput deviceInputWithDevice:audioDevice error:&error];
//  if (!audioDeviceInput)
//  {
//    CAMERA_SETUP_FAILED
//    logComputeError("Could not create audio device input: %s", error);
//  }
//
//  if ([staticSession canAddInput:audioDeviceInput])
//  {
//    [staticSession addInput:audioDeviceInput];
//  }
//  else
//  {
//    CAMERA_SETUP_FAILED
//    logComputeError("Could not add audio device input to the session");
//  }

  staticVideoDataOutput = [[AVCaptureVideoDataOutput alloc] init];
  if ([staticSession canAddOutput:staticVideoDataOutput])
  {
    [staticSession addOutput:staticVideoDataOutput];
    [staticVideoDataOutput setAlwaysDiscardsLateVideoFrames:YES];

    logComputeMessage("Supported camera video formats:");
    for (const NSNumber* format : [staticVideoDataOutput availableVideoCVPixelFormatTypes])
    {
      logComputeMessage("%c%c%c%c %d", (format.intValue>>24)&255, (format.intValue>>16)&255, (format.intValue>>8)&255, format.intValue&255, format.intValue);
    }

    [staticVideoDataOutput setVideoSettings:[NSDictionary dictionaryWithObjects:@[[NSNumber numberWithInt:kCVPixelFormatType_32BGRA], @YES]
                                                                        forKeys:@[(id)kCVPixelBufferPixelFormatTypeKey, (id)kCVPixelBufferMetalCompatibilityKey]]];
    [staticVideoDataOutput setSampleBufferDelegate:[[SampleBufferDeligate alloc] init] queue:staticSessionQueue];
#if TARGET_OS_IPHONE
    [staticVideoDataOutput setAutomaticallyConfiguresOutputBufferDimensions:YES];
#endif
  }
  else
  {
    CAMERA_SETUP_FAILED
    logComputeError("Could not add video data output to the session");
  }

  [staticSession commitConfiguration];
  [staticSession startRunning];
}

bool CameraInterface::isActive()const
{
  return staticCameraSetupResult == AVCamSetupResultSuccess && staticWidth != 0;
}

const ComputeTexture* CameraInterface::getCurrentFrame()const
{
  if (!isActive())
    return NULL;

  staticMetalTexture = ComputeTexture(staticMetalTextures[staticMetalTextureIndex], (uint[2]){staticWidth, staticHeight}, staticBytesPerPixel);
  return &staticMetalTexture;
}

#else

CameraInterface::CameraInterface(ComputeInterface* compute)
{
  logComputeError("CameraInterface is not supported with OpenCL");
}

CameraInterface::~CameraInterface()
{}

// Call this on the session queue.
void CameraInterface::startSession()
{}

const ComputeTexture* CameraInterface::getCurrentFrame()const
{
  return NULL;
}

bool CameraInterface::isActive()const
{
  return false;
}

#endif

uint CameraInterface::width()const
{
  return staticWidth;
}

uint CameraInterface::height()const
{
  return staticHeight;
}

uint CameraInterface::bytesPerPixel()const
{
  return staticBytesPerPixel;
}
