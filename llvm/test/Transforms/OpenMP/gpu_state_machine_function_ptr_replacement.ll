; RUN: opt -S -passes=openmpopt -pass-remarks=openmp-opt -openmp-print-gpu-kernels < %s | FileCheck %s
; RUN: opt -S        -openmpopt -pass-remarks=openmp-opt -openmp-print-gpu-kernels < %s | FileCheck %s

; C input used for this test:

; void bar(void) {
;     #pragma omp parallel
;     { }
; }
; void foo(void) {
;   #pragma omp target teams
;   {
;     #pragma omp parallel
;     {}
;     bar();
;     #pragma omp parallel
;     {}
;   }
; }

; Verify we replace the function pointer uses for the first and last outlined
; region (1 and 3) but not for the middle one (2) because it could be called from
; another kernel.

; CHECK-DAG: @__omp_outlined__1_wrapper.ID = private constant i8 undef
; CHECK-DAG: @__omp_outlined__3_wrapper.ID = private constant i8 undef

; CHECK-DAG:   icmp eq i8* %5, @__omp_outlined__1_wrapper.ID
; CHECK-DAG:   icmp eq i8* %7, @__omp_outlined__3_wrapper.ID

; CHECK-DAG:   call void @__kmpc_kernel_prepare_parallel(i8* @__omp_outlined__1_wrapper.ID)
; CHECK-DAG:   call void @__kmpc_kernel_prepare_parallel(i8* @__omp_outlined__3_wrapper.ID)


%struct.ident_t = type { i32, i32, i32, i32, i8* }

@.str = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@0 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str, i32 0, i32 0) }, align 8

define internal void @__omp_offloading_35_a1e179_foo_l7_worker() {
entry:
  %work_fn = alloca i8*, align 8
  %exec_status = alloca i8, align 1
  store i8* null, i8** %work_fn, align 8
  store i8 0, i8* %exec_status, align 1
  br label %.await.work

.await.work:                                      ; preds = %.barrier.parallel, %entry
  call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  %0 = call i1 @__kmpc_kernel_parallel(i8** %work_fn)
  %1 = zext i1 %0 to i8
  store i8 %1, i8* %exec_status, align 1
  %2 = load i8*, i8** %work_fn, align 8
  %should_terminate = icmp eq i8* %2, null
  br i1 %should_terminate, label %.exit, label %.select.workers

.select.workers:                                  ; preds = %.await.work
  %3 = load i8, i8* %exec_status, align 1
  %is_active = icmp ne i8 %3, 0
  br i1 %is_active, label %.execute.parallel, label %.barrier.parallel

.execute.parallel:                                ; preds = %.select.workers
  %4 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @0)
  %5 = load i8*, i8** %work_fn, align 8
  %work_match = icmp eq i8* %5, bitcast (void (i16, i32)* @__omp_outlined__1_wrapper to i8*)
  br i1 %work_match, label %.execute.fn, label %.check.next

.execute.fn:                                      ; preds = %.execute.parallel
  call void @__omp_outlined__1_wrapper(i16 0, i32 %4)
  br label %.terminate.parallel

.check.next:                                      ; preds = %.execute.parallel
  %6 = load i8*, i8** %work_fn, align 8
  %work_match1 = icmp eq i8* %6, bitcast (void (i16, i32)* @__omp_outlined__2_wrapper to i8*)
  br i1 %work_match1, label %.execute.fn2, label %.check.next3

.execute.fn2:                                     ; preds = %.check.next
  call void @__omp_outlined__2_wrapper(i16 0, i32 %4)
  br label %.terminate.parallel

.check.next3:                                     ; preds = %.check.next
  %7 = load i8*, i8** %work_fn, align 8
  %work_match4 = icmp eq i8* %7, bitcast (void (i16, i32)* @__omp_outlined__3_wrapper to i8*)
  br i1 %work_match4, label %.execute.fn5, label %.check.next6

.execute.fn5:                                     ; preds = %.check.next3
  call void @__omp_outlined__3_wrapper(i16 0, i32 %4)
  br label %.terminate.parallel

.check.next6:                                     ; preds = %.check.next3
  %8 = bitcast i8* %2 to void (i16, i32)*
  call void %8(i16 0, i32 %4)
  br label %.terminate.parallel

.terminate.parallel:                              ; preds = %.check.next6, %.execute.fn5, %.execute.fn2, %.execute.fn
  call void @__kmpc_kernel_end_parallel()
  br label %.barrier.parallel

.barrier.parallel:                                ; preds = %.terminate.parallel, %.select.workers
  call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  br label %.await.work

.exit:                                            ; preds = %.await.work
  ret void
}

define weak void @__omp_offloading_35_a1e179_foo_l7() {
entry:
  %.zero.addr = alloca i32, align 4
  %.threadid_temp. = alloca i32, align 4
  store i32 0, i32* %.zero.addr, align 4
  %nvptx_warp_size = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  %nvptx_num_threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %thread_limit = sub nuw i32 %nvptx_num_threads, %nvptx_warp_size
  %nvptx_tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %0 = icmp ult i32 %nvptx_tid, %thread_limit
  br i1 %0, label %.worker, label %.mastercheck

.worker:                                          ; preds = %entry
  call void @__omp_offloading_35_a1e179_foo_l7_worker()
  br label %.exit

.mastercheck:                                     ; preds = %entry
  %nvptx_num_threads1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %nvptx_warp_size2 = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  %1 = sub nuw i32 %nvptx_warp_size2, 1
  %2 = xor i32 %1, -1
  %3 = sub nuw i32 %nvptx_num_threads1, 1
  %master_tid = and i32 %3, %2
  %nvptx_tid3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %4 = icmp eq i32 %nvptx_tid3, %master_tid
  br i1 %4, label %.master, label %.exit

.master:                                          ; preds = %.mastercheck
  %nvptx_warp_size4 = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  %nvptx_num_threads5 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %thread_limit6 = sub nuw i32 %nvptx_num_threads5, %nvptx_warp_size4
  call void @__kmpc_kernel_init(i32 %thread_limit6, i16 1)
  call void @__kmpc_data_sharing_init_stack()
  %5 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @0)
  store i32 %5, i32* %.threadid_temp., align 4
  call void @__omp_outlined__(i32* %.threadid_temp., i32* %.zero.addr)
  br label %.termination.notifier

.termination.notifier:                            ; preds = %.master
  call void @__kmpc_kernel_deinit(i16 1)
  call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  br label %.exit

.exit:                                            ; preds = %.termination.notifier, %.mastercheck, %.worker
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.warpsize()

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

define internal void @__omp_outlined__(i32* noalias %.global_tid., i32* noalias %.bound_tid.) {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %.zero.addr = alloca i32, align 4
  %.zero.addr1 = alloca i32, align 4
  store i32 0, i32* %.zero.addr1, align 4
  store i32 0, i32* %.zero.addr, align 4
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  call void @__kmpc_kernel_prepare_parallel(i8* bitcast (void (i16, i32)* @__omp_outlined__1_wrapper to i8*))
  call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  call void @bar()
  call void @__kmpc_kernel_prepare_parallel(i8* bitcast (void (i16, i32)* @__omp_outlined__3_wrapper to i8*))
  call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  ret void
}

define internal void @__omp_outlined__1(i32* noalias %.global_tid., i32* noalias %.bound_tid.) {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  ret void
}

define internal void @__omp_outlined__1_wrapper(i16 zeroext %0, i32 %1) {
entry:
  %.addr = alloca i16, align 2
  %.addr1 = alloca i32, align 4
  %.zero.addr = alloca i32, align 4
  %global_args = alloca i8**, align 8
  store i32 0, i32* %.zero.addr, align 4
  store i16 %0, i16* %.addr, align 2
  store i32 %1, i32* %.addr1, align 4
  call void @__kmpc_get_shared_variables(i8*** %global_args)
  call void @__omp_outlined__1(i32* %.addr1, i32* %.zero.addr)
  ret void
}

define hidden void @bar() {
entry:
  %.zero.addr = alloca i32, align 4
  store i32 0, i32* %.zero.addr, align 4
  call void @__kmpc_kernel_prepare_parallel(i8* bitcast (void (i16, i32)* @__omp_outlined__2_wrapper to i8*))
  call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  ret void
}

define internal void @__omp_outlined__2(i32* noalias %.global_tid., i32* noalias %.bound_tid.) {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  ret void
}

define internal void @__omp_outlined__2_wrapper(i16 zeroext %0, i32 %1) {
entry:
  %.addr = alloca i16, align 2
  %.addr1 = alloca i32, align 4
  %.zero.addr = alloca i32, align 4
  %global_args = alloca i8**, align 8
  store i32 0, i32* %.zero.addr, align 4
  store i16 %0, i16* %.addr, align 2
  store i32 %1, i32* %.addr1, align 4
  call void @__kmpc_get_shared_variables(i8*** %global_args)
  call void @__omp_outlined__2(i32* %.addr1, i32* %.zero.addr)
  ret void
}

define internal void @__omp_outlined__3(i32* noalias %.global_tid., i32* noalias %.bound_tid.) {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  ret void
}

define internal void @__omp_outlined__3_wrapper(i16 zeroext %0, i32 %1) {
entry:
  %.addr = alloca i16, align 2
  %.addr1 = alloca i32, align 4
  %.zero.addr = alloca i32, align 4
  %global_args = alloca i8**, align 8
  store i32 0, i32* %.zero.addr, align 4
  store i16 %0, i16* %.addr, align 2
  store i32 %1, i32* %.addr1, align 4
  call void @__kmpc_get_shared_variables(i8*** %global_args)
  call void @__omp_outlined__3(i32* %.addr1, i32* %.zero.addr)
  ret void
}

declare void @__kmpc_data_sharing_init_stack()

declare void @__kmpc_get_shared_variables(i8*** nocapture %GlobalArgs)

declare void @__kmpc_kernel_init(i32 %ThreadLimit, i16 signext %RequiresOMPRuntime)

declare void @__kmpc_kernel_deinit(i16 signext %IsOMPRuntimeInitialized)

declare void @__kmpc_kernel_prepare_parallel(i8* %WorkFn)

declare zeroext i1 @__kmpc_kernel_parallel(i8** nocapture %WorkFn)

declare void @__kmpc_kernel_end_parallel()

declare void @__kmpc_barrier_simple_spmd(%struct.ident_t* nocapture readnone %loc_ref, i32 %tid)

declare i32 @__kmpc_global_thread_num(%struct.ident_t* nocapture readnone)


!llvm.module.flags = !{!0, !1, !2, !3}
!omp_offload.info = !{!4}
!nvvm.annotations = !{!5, !6, !7, !6, !8, !8, !8, !8, !9, !9, !8, !6, !7, !6, !8, !8, !8, !8, !9, !9, !8, !6, !7, !6, !8, !8, !8, !8, !9, !9, !8, !6, !7, !6, !8, !8, !8, !8, !9, !9, !8, !6, !7, !6, !8, !8, !8, !8, !9, !9, !8, !6, !7, !6, !8, !8, !8, !8, !9, !9, !8, !6, !7, !6, !8, !8, !8, !8, !9, !9, !8, !6, !7, !6, !8, !8, !8, !8, !9, !9, !8, !6, !7, !6, !8, !8, !8, !8, !9, !9, !8, !6, !7, !6, !8, !8, !8, !8, !9, !9, !8, !6, !7, !6, !8, !8, !8, !8, !9, !9, !8, !6, !7, !6, !8, !8, !8, !8, !9, !9, !8, !6, !7, !6, !8, !8, !8, !8, !9, !9, !8}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 2]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"PIC Level", i32 2}
!3 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!4 = !{i32 0, i32 53, i32 10609017, !"foo", i32 7, i32 0}
!5 = !{void ()* @__omp_offloading_35_a1e179_foo_l7, !"kernel", i32 1}
!6 = !{null, !"align", i32 8}
!7 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!8 = !{null, !"align", i32 16}
!9 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
