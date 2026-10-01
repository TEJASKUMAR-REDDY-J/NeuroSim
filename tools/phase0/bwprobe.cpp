// minimal OpenCL bandwidth + launch-latency probe
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <chrono>
#include <cstdio>
#include <vector>
static const char* src = R"(
kernel void copy4(global const float4* a, global float4* b) { const size_t i=get_global_id(0); b[i]=a[i]; }
kernel void read4(global const float4* a, global float* b) { const size_t i=get_global_id(0); float4 v=a[i]; if(v.x==-1.2345f) b[0]=v.y; }
kernel void triad4(global const float4* a, global const float4* c, global float4* b) { const size_t i=get_global_id(0); b[i]=a[i]+1.5f*c[i]; }
kernel void empty(global float* b) { }
)";
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
	cl_platform_id p; cl_device_id d; cl_int e;
	clGetPlatformIDs(1, &p, nullptr); clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 1, &d, nullptr);
	char name[256]; clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(name), name, nullptr);
	cl_context ctx = clCreateContext(nullptr, 1, &d, nullptr, nullptr, &e);
	cl_command_queue q = clCreateCommandQueueWithProperties(ctx, d, nullptr, &e);
	cl_program pr = clCreateProgramWithSource(ctx, 1, &src, nullptr, &e); clBuildProgram(pr, 1, &d, "", nullptr, nullptr);
	const size_t n4 = 32u<<20; // 32M float4 = 512 MB per buffer
	cl_mem a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n4*16, nullptr, &e), b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n4*16, nullptr, &e), c = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n4*16, nullptr, &e);
	float z = 0.0f; clEnqueueFillBuffer(q, a, &z, 4, 0, n4*16, 0, nullptr, nullptr); clEnqueueFillBuffer(q, b, &z, 4, 0, n4*16, 0, nullptr, nullptr); clEnqueueFillBuffer(q, c, &z, 4, 0, n4*16, 0, nullptr, nullptr); clFinish(q);
	printf("device: %s\n", name);
	struct K { const char* n; int args; double bytes; } ks[] = { {"copy4", 2, 32.0}, {"read4", 2, 16.0}, {"triad4", 3, 48.0} };
	for(auto& k : ks) {
		cl_kernel kk = clCreateKernel(pr, k.n, &e);
		clSetKernelArg(kk, 0, sizeof(cl_mem), &a); clSetKernelArg(kk, 1, sizeof(cl_mem), k.args==3 ? &c : &b); if(k.args==3) clSetKernelArg(kk, 2, sizeof(cl_mem), &b);
		size_t g = n4, l = 64; clEnqueueNDRangeKernel(q, kk, 1, nullptr, &g, &l, 0, nullptr, nullptr); clFinish(q);
		double best = 1e30; for(int r=0; r<10; r++) { double t0=now(); clEnqueueNDRangeKernel(q, kk, 1, nullptr, &g, &l, 0, nullptr, nullptr); clFinish(q); best = std::min(best, now()-t0); }
		printf("%-7s %.2f GB/s\n", k.n, k.bytes*(double)n4/best*1e-9);
	}
	cl_kernel ke = clCreateKernel(pr, "empty", &e); clSetKernelArg(ke, 0, sizeof(cl_mem), &a); size_t g = 64, l = 64;
	for(int r=0; r<100; r++) clEnqueueNDRangeKernel(q, ke, 1, nullptr, &g, &l, 0, nullptr, nullptr); clFinish(q);
	const int R = 2000; double t0 = now(); for(int r=0; r<R; r++) { clEnqueueNDRangeKernel(q, ke, 1, nullptr, &g, &l, 0, nullptr, nullptr); clFinish(q); } const double sync_us = (now()-t0)/R*1e6;
	t0 = now(); for(int r=0; r<R; r++) clEnqueueNDRangeKernel(q, ke, 1, nullptr, &g, &l, 0, nullptr, nullptr); clFinish(q); const double async_us = (now()-t0)/R*1e6;
	printf("empty kernel: %.1f us/launch with finish, %.1f us/launch batched\n", sync_us, async_us);
	return 0;
}
