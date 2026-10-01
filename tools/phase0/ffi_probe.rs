// same empty-kernel launch test as bw.cpp, but driven from Rust through raw FFI
use std::ffi::c_void;
use std::ptr::null_mut;
use std::time::Instant;
type H = *mut c_void;
#[link(name = "OpenCL")]
extern "system" {
    fn clGetPlatformIDs(n: u32, p: *mut H, np: *mut u32) -> i32;
    fn clGetDeviceIDs(p: H, t: u64, n: u32, d: *mut H, nd: *mut u32) -> i32;
    fn clCreateContext(props: *const isize, n: u32, d: *const H, cb: *const c_void, ud: *mut c_void, e: *mut i32) -> H;
    fn clCreateCommandQueueWithProperties(c: H, d: H, props: *const u64, e: *mut i32) -> H;
    fn clCreateProgramWithSource(c: H, n: u32, s: *const *const u8, l: *const usize, e: *mut i32) -> H;
    fn clBuildProgram(p: H, n: u32, d: *const H, o: *const u8, cb: *const c_void, ud: *mut c_void) -> i32;
    fn clCreateKernel(p: H, name: *const u8, e: *mut i32) -> H;
    fn clCreateBuffer(c: H, f: u64, s: usize, h: *mut c_void, e: *mut i32) -> H;
    fn clSetKernelArg(k: H, i: u32, s: usize, v: *const c_void) -> i32;
    fn clEnqueueNDRangeKernel(q: H, k: H, dim: u32, off: *const usize, g: *const usize, l: *const usize, ne: u32, ev: *const H, e: *mut H) -> i32;
    fn clFinish(q: H) -> i32;
}
fn main() {
    unsafe {
        let (mut p, mut d, mut e): (H, H, i32) = (null_mut(), null_mut(), 0);
        clGetPlatformIDs(1, &mut p, null_mut());
        clGetDeviceIDs(p, 0xFFFF_FFFF, 1, &mut d, null_mut());
        let ctx = clCreateContext(std::ptr::null(), 1, &d, std::ptr::null(), null_mut(), &mut e);
        let q = clCreateCommandQueueWithProperties(ctx, d, std::ptr::null(), &mut e);
        let src = b"kernel void empty(global float* b) { }\0";
        let sp = src.as_ptr();
        let pr = clCreateProgramWithSource(ctx, 1, &sp, std::ptr::null(), &mut e);
        clBuildProgram(pr, 1, &d, b"\0".as_ptr(), std::ptr::null(), null_mut());
        let k = clCreateKernel(pr, b"empty\0".as_ptr(), &mut e);
        let b = clCreateBuffer(ctx, 1, 1024, null_mut(), &mut e);
        clSetKernelArg(k, 0, std::mem::size_of::<H>(), &b as *const H as *const c_void);
        let (g, l) = (64usize, 64usize);
        let launch = || clEnqueueNDRangeKernel(q, k, 1, std::ptr::null(), &g, &l, 0, std::ptr::null(), null_mut());
        for _ in 0..100 { launch(); }
        clFinish(q);
        let r = 2000;
        let t = Instant::now(); for _ in 0..r { launch(); clFinish(q); }
        let sync_us = t.elapsed().as_secs_f64() / r as f64 * 1e6;
        let t = Instant::now(); for _ in 0..r { launch(); } clFinish(q);
        let async_us = t.elapsed().as_secs_f64() / r as f64 * 1e6;
        println!("rust ffi empty kernel: {:.1} us/launch with finish, {:.1} us/launch batched", sync_us, async_us);
    }
}
