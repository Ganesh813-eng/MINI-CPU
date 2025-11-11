# -----------------------------------------------------------
# MiniCPU Simulator & Debugger — Streamlit (from scratch)
# -----------------------------------------------------------
# Run:
#   pip install streamlit
#   streamlit run minicpu_streamlit.py
#
# Features:
# - CPU core with 8 registers (R0..R7, where R7=SP), 64KB memory
# - Fetch/Decode/Execute for a small ISA
# - Labels + jumps, stack ops, breakpoints
# - Step / Run / Reset; memory inspector; sample programs
# - Upload/Download program text
#
# Instruction Set (operands: Rn, imm, [addr], [label], label)
#   NOP
#   HALT
#   MOV   Rn, (Rm|imm)
#   LOAD  Rn, [addr|label]
#   STORE Rn, [addr|label]
#   ADD   Rn, (Rm|imm)
#   SUB   Rn, (Rm|imm)
#   CMP   (Rn|imm), (Rm|imm)
#   JMP   label|imm
#   JZ    label|imm      ; jump if ZF==1
#   JNZ   label|imm      ; jump if ZF==0
#   PUSH  Rn
#   POP   Rn
#   CALL  label|imm
#   RET
#
# -----------------------------------------------------------

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import streamlit as st

# --------------------------
# CPU CORE
# --------------------------
@dataclass
class MiniCPU:
    regs: List[int] = field(default_factory=lambda: [0]*8)        # R0..R7 (R7=SP)
    pc: int = 0                                                   # Program Counter (index into program list)
    zf: int = 0                                                   # Zero Flag
    memory: List[int] = field(default_factory=lambda: [0]*65536)  # 64 KB
    program: List[Tuple[str, Tuple]] = field(default_factory=list) # [(op, args_tuple)]
    labels: Dict[str, int] = field(default_factory=dict)           # label -> instr index (pc)
    halted: bool = False
    breakpoints: set = field(default_factory=set)
    last_status: str = ""

    # ------------- Core API -------------
    def reset_state(self):
        self.regs = [0]*8
        self.pc = 0
        self.zf = 0
        self.memory = [0]*65536
        self.halted = False
        self.breakpoints = set()
        self.last_status = ""

    def load_asm(self, text: str) -> None:
        """Parse assembly text, build program + labels."""
        self.reset_state()
        self.program = []
        self.labels = {}

        # Strip comments (# or ;) and empty lines
        lines: List[str] = []
        for raw in text.splitlines():
            line = re.split(r"[#;]", raw, 1)[0].strip()
            if line:
                lines.append(line)

        # First pass: collect labels
        pc_counter = 0
        for ln in lines:
            if ln.endswith(":"):
                label = ln[:-1].strip()
                if not label:
                    raise ValueError("Empty label defined.")
                if label in self.labels:
                    raise ValueError(f"Duplicate label: {label}")
                self.labels[label] = pc_counter
            else:
                pc_counter += 1

        # Second pass: instructions
        for ln in lines:
            if ln.endswith(":"):
                continue
            op, *rest = re.split(r"\s+", ln, maxsplit=1)
            op = op.upper().strip()
            args: List[Tuple[str, int|str]] = []
            if rest:
                parts = [p.strip() for p in rest[0].split(",")]
                for p in parts:
                    # Register
                    if re.fullmatch(r"R[0-7]", p.upper()):
                        args.append(("reg", int(p[1])))
                        continue
                    # Immediate integer (signed)
                    if re.fullmatch(r"-?\d+", p):
                        args.append(("imm", int(p)))
                        continue
                    # Memory reference [addr or label]
                    if re.fullmatch(r"\[.+\]", p):
                        inner = p[1:-1].strip()
                        if re.fullmatch(r"-?\d+", inner):
                            args.append(("maddr", int(inner)))
                        else:
                            args.append(("mlabel", inner))
                        continue
                    # Label (for jumps/calls)
                    args.append(("label", p))
            self.program.append((op, tuple(args)))
        self.last_status = "Program loaded"

    # ---------- Helpers ----------
    def _get_val(self, arg: Tuple[str, int|str]) -> int:
        kind, v = arg
        if kind == "reg":
            return self.regs[int(v)]
        if kind == "imm":
            return int(v)
        if kind == "maddr":
            addr = int(v) & 0xFFFF
            return self.memory[addr] & 0xFFFF
        if kind == "mlabel":
            addr = self.labels.get(str(v), 0)
            return self.memory[addr] & 0xFFFF
        if kind == "label":
            return self.labels.get(str(v), 0)
        raise ValueError(f"Bad operand: {arg}")

    def _set_reg(self, idx: int, val: int) -> None:
        self.regs[idx] = val & 0xFFFF
        self.zf = int(self.regs[idx] == 0)

    # ---------- Execute one instruction ----------
    def step(self) -> str:
        if self.halted or not (0 <= self.pc < len(self.program)):
            self.halted = True
            self.last_status = "HALT"
            return "HALT"

        if self.pc in self.breakpoints:
            self.last_status = f"BREAK @ {self.pc}"
            return "BREAK"

        op, args = self.program[self.pc]
        self.pc += 1  # optimistic PC advance

        r = lambda a: a[1]  # reg index accessor
        gv = lambda a: self._get_val(a)

        try:
            if op == "NOP":
                pass
            elif op == "HALT":
                self.halted = True

            elif op == "MOV":
                self._set_reg(int(r(args[0])), gv(args[1]))

            elif op == "LOAD":
                self._set_reg(int(r(args[0])), gv(args[1]))

            elif op == "STORE":
                kind, v = args[1]
                src = int(r(args[0]))
                if kind == "maddr":
                    addr = int(v) & 0xFFFF
                else:
                    addr = self.labels.get(str(v), 0)
                self.memory[addr] = self.regs[src] & 0xFFFF

            elif op == "ADD":
                dst = int(r(args[0]))
                self._set_reg(dst, self.regs[dst] + gv(args[1]))

            elif op == "SUB":
                dst = int(r(args[0]))
                self._set_reg(dst, self.regs[dst] - gv(args[1]))

            elif op == "CMP":
                a = gv(args[0])
                b = gv(args[1])
                self.zf = int((a & 0xFFFF) == (b & 0xFFFF))

            elif op == "JMP":
                self.pc = gv(args[0])

            elif op == "JZ":
                if self.zf == 1:
                    self.pc = gv(args[0])

            elif op == "JNZ":
                if self.zf == 0:
                    self.pc = gv(args[0])

            elif op == "PUSH":
                reg_idx = int(r(args[0]))
                self.regs[7] = (self.regs[7] - 1) & 0xFFFF
                self.memory[self.regs[7]] = self.regs[reg_idx] & 0xFFFF

            elif op == "POP":
                reg_idx = int(r(args[0]))
                self._set_reg(reg_idx, self.memory[self.regs[7]])
                self.regs[7] = (self.regs[7] + 1) & 0xFFFF

            elif op == "CALL":
                # push return address (current PC), jump to target
                self.regs[7] = (self.regs[7] - 1) & 0xFFFF
                self.memory[self.regs[7]] = self.pc & 0xFFFF
                self.pc = gv(args[0])

            elif op == "RET":
                self.pc = self.memory[self.regs[7]] & 0xFFFF
                self.regs[7] = (self.regs[7] + 1) & 0xFFFF

            else:
                self.last_status = f"Unknown op: {op}"
                return self.last_status

            self.last_status = op
            return op

        except Exception as e:
            self.halted = True
            self.last_status = f"ERROR: {e}"
            return self.last_status

    # ---------- Run until halt or breakpoint or max steps ----------
    def run(self, max_steps: int = 10000) -> str:
        steps = 0
        while not self.halted and steps < max_steps:
            status = self.step()
            if status.startswith("BREAK") or status.startswith("ERROR"):
                return status
            steps += 1
        if steps >= max_steps:
            self.last_status = "MAX_STEPS_REACHED"
        return self.last_status


# --------------------------
# Sample programs
# --------------------------
SAMPLES: Dict[str, str] = {
    "Sum 1..5 to R0 (stores at [100])": """\
        MOV R0, 0
        MOV R1, 1
loop:   ADD R0, R1
        ADD R1, 1
        CMP R1, 6
        JNZ loop
        STORE R0, [100]
        HALT
""",
    "Factorial (5) using stack": """\
        MOV R7, 500      ; init SP near 500
        MOV R0, 1        ; result
        MOV R1, 5        ; n
fact:   CMP R1, 1
        JZ end
        PUSH R1
        MUL:             ; emulate R0 *= R1 via loop add
        MOV R2, 0
mulL:   CMP R2, R1
        JZ mulE
        ADD R2, 1
        ADD R0, 1
        JMP mulL
mulE:   POP R1
        SUB R1, 1
        JMP fact
end:    STORE R0, [120]
        HALT
""",
    "CALL/RET demo": """\
        MOV R7, 1024
        MOV R0, 10
        CALL inc
        CALL inc
        STORE R0, [200]
        HALT
inc:    ADD R0, 1
        RET
"""
}

HELP_TEXT = """
**Instruction Set Quick Help**

- `MOV Rn, X` — move immediate/register to Rn  
- `LOAD Rn, [addr|label]` — load from memory to Rn  
- `STORE Rn, [addr|label]` — store Rn to memory  
- `ADD/SUB Rn, X` — arithmetic, updates ZF via `_set_reg`  
- `CMP A, B` — sets ZF=1 if A==B else 0  
- `JMP lbl|imm`, `JZ lbl|imm`, `JNZ lbl|imm` — control flow  
- `PUSH/POP Rn` — stack ops, **R7 is SP**  
- `CALL lbl|imm` / `RET` — subroutines  
- `HALT` — stop execution  
- Memory size: **64KB** (0..65535). Values are 16-bit (wrap).
"""

# --------------------------
# Streamlit App
# --------------------------
st.set_page_config(page_title="MiniCPU – Streamlit Debugger", layout="wide")

# One CPU per session
if "cpu" not in st.session_state:
    st.session_state.cpu = MiniCPU()
cpu: MiniCPU = st.session_state.cpu

# --- Sidebar ---
st.sidebar.title("⚙️ Controls")
sample_choice = st.sidebar.selectbox("Load Sample Program", ["(None)"] + list(SAMPLES.keys()))
if st.sidebar.button("Load Sample"):
    if sample_choice != "(None)":
        st.session_state.program_text = SAMPLES[sample_choice]
        st.toast(f"Loaded sample: {sample_choice}")

max_steps = st.sidebar.number_input("Run max steps", 10, 100000, 2000, step=10)
bp_to_add = st.sidebar.number_input("Breakpoint (PC index)", 0, 100000, 0, step=1)
c1, c2 = st.sidebar.columns(2)
if c1.button("Add Breakpoint"):
    cpu.breakpoints.add(int(bp_to_add))
    st.toast(f"Breakpoint added at PC={bp_to_add}")
if c2.button("Remove Breakpoint"):
    cpu.breakpoints.discard(int(bp_to_add))
    st.toast(f"Breakpoint removed at PC={bp_to_add}")

if st.sidebar.button("Reset CPU"):
    cpu.reset_state()
    st.toast("CPU state reset")

st.sidebar.markdown("---")
st.sidebar.markdown(HELP_TEXT)

# --- Main layout ---
left, right = st.columns([1.15, 1])

# Program editor
with left:
    st.title("🧠 MiniCPU Simulator & Debugger")
    default_prog = SAMPLES["Sum 1..5 to R0 (stores at [100])"]
    program_text = st.text_area("Assembly Program (labels end with ':')",
                                value=st.session_state.get("program_text", default_prog),
                                height=280, key="program_text_area")

    cc1, cc2, cc3, cc4 = st.columns(4)
    if cc1.button("Load Program"):
        try:
            cpu.load_asm(program_text)
            st.toast("Program loaded ✅")
        except Exception as e:
            st.error(f"Assembler error: {e}")

    if cc2.button("Step"):
        status = cpu.step()
        st.toast(f"Step: {status}")

    if cc3.button("Run"):
        status = cpu.run(int(max_steps))
        st.toast(f"Run: {status}")

    if cc4.button("HALT"):
        cpu.halted = True
        cpu.last_status = "HALT"
        st.toast("Halted")

    st.markdown("##### ⛳ Breakpoints")
    st.write(sorted(cpu.breakpoints))

    st.markdown("##### ⬇️ Download / ⬆️ Upload")
    st.download_button("Download Program", program_text, file_name="program.asm", mime="text/plain")

    uploaded = st.file_uploader("Upload .asm file", type=["asm", "txt"])
    if uploaded:
        st.session_state.program_text = uploaded.read().decode("utf-8")
        st.rerun()

# State & Memory view
with right:
    st.subheader("CPU State")
    st.write(f"**PC**: {cpu.pc}  |  **ZF**: {cpu.zf}  |  **HALTED**: {cpu.halted}  |  **Last**: {cpu.last_status}")

    st.subheader("Registers (R0..R7, R7 = SP)")
    reg_table = {f"R{i}": cpu.regs[i] for i in range(8)}
    st.table({"Register": list(reg_table.keys()), "Value": list(reg_table.values())})

    st.subheader("Memory Inspector")
    mcol1, mcol2 = st.columns(2)
    start = mcol1.number_input("Start address", min_value=0, max_value=65535, value=96, step=1)
    count = mcol2.number_input("Count", min_value=1, max_value=256, value=32, step=1)

    start_i = int(start)
    end_i = min(65536, start_i + int(count))
    addrs = list(range(start_i, end_i))
    vals = [cpu.memory[a] for a in addrs]
    st.table({"Addr": addrs, "Val": vals})

    st.caption("Tip: Use the **PC** as a quick guide to your current instruction index. Breakpoints halt before execution at that PC.")

# Footer
st.markdown("---")
st.caption("MiniCPU Streamlit — clean rewrite • Supports: NOP, HALT, MOV, LOAD, STORE, ADD, SUB, CMP, JMP, JZ, JNZ, PUSH, POP, CALL, RET • R7 is SP • 16-bit data, 64KB memory.")

