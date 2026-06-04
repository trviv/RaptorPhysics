Raptor Physics
==============

Raptor Physics is a GPGPU based Particle Physics Simulator and Ray Tracer created for learning purposes.
It supports OpenCL and Metal compute backends, and has performant implementations of core compute functionality such as Prefix Scan, Reduce, and Radix Sort.
It's developed from scratch and has no dependency on any pre-existing libraries, making it suitable for any platform.

Papers implemented in this code base include:

**Particle physics**
 * Macklin, M., Müller, M., Chentanez, N., Kim, T.-Y. 2014. *Unified Particle Physics for Real-Time Applications.* ACM Trans. Graph. 33(4) (SIGGRAPH 2014).
 * Macklin, M., Müller, M. 2013. *Position Based Fluids.* ACM Trans. Graph. 32(4) (SIGGRAPH 2013).
 * Solenthaler, B., Pajarola, R. 2009. *Predictive-Corrective Incompressible SPH.* ACM Trans. Graph. 28(3) (SIGGRAPH 2009).
 * Bridson, R., Fedkiw, R., Anderson, J. 2002. *Robust Treatment of Collisions, Contact and Friction for Cloth Animation* (constraint averaging). ACM Trans. Graph. 21(3) (SIGGRAPH 2002).
 * Tonge, R., Benevolenski, F., Voroshilov, A. 2012. *Mass Splitting for Jitter-Free Parallel Rigid Body Simulation.* ACM Trans. Graph. 31(4) (SIGGRAPH 2012).
 * Higham, N. J. 1986. *Computing the Polar Decomposition with Applications.* SIAM J. Sci. Stat. Comput. 7(4), 1160–1174.

**Ray tracing**
 * Hapala, M., Davidovič, T., Wald, I., Havran, V., Slusallek, P. 2011. *Efficient Stack-less BVH Traversal for Ray Tracing.* Spring Conference on Computer Graphics (SCCG 2011).
 * Karras, T. 2012. *Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees.* High-Performance Graphics (HPG 2012).
 * Meister, D., Bittner, J. 2018. *Parallel Locally-Ordered Clustering for Bounding Volume Hierarchy Construction* (PLOC). IEEE Trans. Vis. Comput. Graph. 24(3), 1345–1353.

**Parallel primitives**
 * Hillis, W. D., Steele, G. L. Jr. 1986. *Data Parallel Algorithms.* Communications of the ACM 29(12), 1170–1183.
 * Yan, S., Long, G., Zhang, Y. 2013. *StreamScan: Fast Scan Algorithms for GPUs without Global Barriers.* PPoPP 2013.
 * Ha, L., Krüger, J., Silva, C. T. 2009. *Fast Four-Way Parallel Radix Sorting on GPUs.* Computer Graphics Forum 28(8), 2368–2378.

## Supported backends
 * Metal
 * OpenCL
 * Vulkan (Pending)

## Supported platforms
 * MacOS & iOS
 * Windows (Pending)
 * Linux (Pending)

## License

Raptor Physics is released under the MIT license, see [LICENSE](./LICENSE).