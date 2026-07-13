import ncu_report

r = ncu_report.load_report("profile/run01_1m_k8192/reports/full_scan_1m_k8192.ncu-rep")
a = r.range_by_idx(0).action_by_idx(0)


def g(n):
    m = a[n]
    return m.value() if m is not None else None


print("duration us:", g("gpu__time_duration.sum"))
print("dram read %:", g("dram__bytes_read.sum.pct_of_peak_sustained_elapsed"))
print("dram write %:", g("dram__bytes_write.sum.pct_of_peak_sustained_elapsed"))
print("issue active %:", g("smsp__issue_active.avg.pct_of_peak_sustained_active"))
print("tensor pipe %:", g("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"))
print("local ld inst:", g("smsp__sass_inst_executed_op_local_ld.sum"))
print("local st inst:", g("smsp__sass_inst_executed_op_local_st.sum"))
print("global st inst:", g("smsp__sass_inst_executed_op_global_st.sum"))
print("global ld inst:", g("smsp__sass_inst_executed_op_global_ld.sum"))
print("atom inst:", g("smsp__sass_inst_executed_op_global_atom.sum"))
print("red inst:", g("smsp__sass_inst_executed_op_global_red.sum"))
print("regs/thread:", g("launch__registers_per_thread"))
print("--- stall cycles per issued inst (>0.5) ---")
rows = []
for n in a.metric_names():
    if "warps_issue_stalled" in n and n.endswith("per_issue_active.ratio"):
        v = a[n].value()
        if v and v > 0.5:
            name = n.split("stalled_")[1].split("_per")[0]
            rows.append((v, name))
for v, name in sorted(rows, reverse=True):
    print(f"  {name:28s} {v:6.2f}")
