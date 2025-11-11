# minicpu_streamlit.py
# Streamlit UI for MiniCPU: Instruction Set Simulator + Debugger
# Run: streamlit run minicpu_streamlit.py

import streamlit as st
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ---------- Core MiniCPU ----------
@dataclass
class MiniCPU:
    regs: List[int] = field(default_factory=lambda: [0]*8)     # R0..R7 (R7 is SP)
    pc: int = 0                                                # Program Counter (index into instructions list)
    zf: int = 0                                                # Zero flag
    memory: List[int] = field(default_factory=lambda: [0]*65536)  # 64KB
    program: List[Tuple[str, Tuple]] = field(default_factory=list)
    labels: Dict[str, int] = field(default_factory=dict)
    halted: bool = False
    breakpoints: set = field(default_factory=set)

    def reset(self):
        self.regs = [0]*8
        self.pc = 0
        self.zf = 0
        self.memory = [0]*65536
        self.halted = False
        self.breakpoints = set()

    # --- Assembly parsing ---
    def load_asm(self, text: str):
        self.reset()
        self.program = []
        self.labels = {}
        lines = []
        for raw in text.splitlines():
            # strip comments (# or ;)
            line = re.split(r"[#;]", raw, 1)[0].strip()
            if not line:
                continue
            lines.append(line)

        # first pass: collect labels
        pc = 0
        for ln in lines:
            if ln.endswith(":"):
                self.labels[ln[:-1].strip()] = pc
            else:
                pc += 1

        # second pass: build instruction list
        for ln in lines:
            if ln.endswith(":"):
                continue
            op, *rest = re.split(r"\s+", ln, maxsplit=1)
            op = op.upper()
            args = []
            if rest:
                # split by commas
                parts = [a.strip() for a in rest[0].split(",")]
                for p in parts:
                    if re.fullmatch(r"R[0-7]", p.upper()):
                        args.append(("reg", int(p[1])))
                    elif re.fullmatch(r"-?\d+", p):
                        args.append(("imm", int(p)))
                    elif re.fullmatch(r"\[.*\]", p):  # [addr] memory literal
                        inner = p[1:-1].strip()
                        if re.fullmatch(r"-?\d+", inner):
                            args.append(("maddr", int(inner)))
                        else:
                            args.append(("mlabel", inner))
                    else:
                        # label (for jumps/calls or mem labels)
                        args.append(("label", p))
            self.program.append((op, tuple(args)))

    def _get_val(self, arg):
        kind, val = arg
        if kind == "reg":
            return self.regs[val]
        if kind == "imm":
            return val
        if kind == "maddr":
            return self.memory[val] & 0xFFFF
        if kind == "mlabel":
            addr = self.labels.get(val, 0)
            return self.memory[addr] & 0xFFFF
        if kind == "label":
            return self.labels.get(val, 0)
        raise ValueError("Bad operand")

    def _set_reg(self, r, v):
        self.regs[r] = v & 0xFFFF
        self.zf = int((self.regs[r] & 0xFFFF) == 0)

    def step(self):
        if self.halted or not (0 <= self.pc < len(self.program)):
            self.halted = True
            return "HALT"

        if self.pc in self.breakpoints:
            return "BREAK"

        op, args = self.program[self.pc]
        self.pc += 1

        def reg_index(a): return a[1]

        if op == "NOP":
            pass
        elif op == "HALT":
            self.halted = True
        elif op == "MOV":  # MOV Rn, Rm|imm
            dst = reg_index(args[0])
            val = self._get_val(args[1])
            self._set_reg(dst, val)
        elif op == "LOAD":  # LOAD Rn, [addr|label]
            dst = reg_index(args[0])
            val = self._get_val(args[1])
            self._set_reg(dst, val)
        elif op == "STORE":  # STORE Rn, [addr|label]
            src = reg_index(args[0])
            kind, v = args[1]
            if kind == "maddr":
                self.memory[v] = self.regs[src] & 0xFFFF
            else:
                addr = self.labels.get(v, 0)
                self.memory[addr] = self.regs[src] & 0xFFFF
        elif op == "ADD":
            dst = reg_index(args[0])
            val = self._get_val(args[1])
            self._set_reg(dst, (self.regs[dst] + val) & 0xFFFF)
        elif op == "SUB":
            dst = reg_index(args[0])
            val = self._get_val(args[1])
            self._set_reg(dst, (self.regs[dst] - val) & 0xFFFF)
        elif op == "CMP":  # CMP Rn, Rm|imm -> set Z if equal
            a = self._get_val(args[0])
            b = self._get_val(args[1])
            self.zf = int((a & 0xFFFF) == (b & 0xFFFF))
        elif op == "JMP":
            self.pc = self._get_val(args[0])
        elif op == "JZ":
            if self.zf == 1:
                self.pc = self._get_val(args[0])
        elif op == "JNZ":
            if self.zf == 0:
                self.pc = self._get_val(args[0])
        elif op == "PUSH":
            r = reg_index(args[0])
            self.regs[7] = (self.regs[7] - 1) & 0xFFFF
            self.memory[self.regs[7]] = self.regs[r] & 0xFFFF
        elif op == "POP":
            r = reg_index(args[0])
            self._set_reg(r, self.memory[self.regs[7]])
            self.regs[7] = (self.regs[7] + 1) & 0xFFFF
        elif op == "CALL":
            # push return address then jump
            self.regs[7] = (self.regs[7] - 1) & 0xFFFF
            self.memory[self.regs[7]] = self.pc & 0xFFFF
            self.pc = self._get_val(args[0])
        elif op == "RET":
            self.pc = self.memory[self.regs[7]] & 0xFFFF
            self.regs[7] = (self.regs[7] + 1) & 0xFFFF
        else:
            return f"Unknown op {op}"
        return op

    def run_steps(self, n: int = 1000):
        steps = 0
        status = ""
        while steps < n and not self.halted:
            if self.pc in self.breakpoints:
                return "BREAK"
            status = self.step()
            steps += 1
        return status or "OK"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="MiniCPU Simulator", layout="wide")

if "cpu" not in st.session_state:
    st.session_state.cpu = MiniCPU()

cpu: MiniCPU = st.session_state.cpu

default_program = """\
# Sample program: sum 1..5 -> R0, store at [100]
        MOV R0, 0
        MOV R1, 1
loop:   ADD R0, R1
        ADD R1, 1
        CMP R1, 6
        JNZ loop
        STORE R0, [100]
        HALT
"""

st.title("MiniCPU: Instruction Set Simulator & Debugger (Streamlit)")

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Assembly Program")
    code = st.text_area("Edit program (labels end with ':')", default_program, height=260)

    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("Load/Reset"):
        cpu.load_asm(code)
        st.success("Program loaded.")
    if c2.button("Step"):
        st.write(cpu.step())
    steps = c3.number_input("Run N steps", min_value=1, max_value=5000, value=50, step=10)
    if c4.button("Run"):
        st.write(cpu.run_steps(int(steps)))
    if c5.button("HALT"):
        cpu.halted = True

    st.markdown("---")
    bcol1, bcol2 = st.columns([1, 1])
    with bcol1:
        st.write("Breakpoints (PC indices): ", sorted(cpu.breakpoints))
        bp_in = st.number_input("Add/Remove breakpoint at PC", min_value=0, value=0)
        b1, b2 = st.columns(2)
        if b1.button("Add BP"):
            cpu.breakpoints.add(int(bp_in))
        if b2.button("Remove BP"):
            cpu.breakpoints.discard(int(bp_in))

with right:
    st.subheader("CPU State")
    reg_rows = {f"R{i}": cpu.regs[i] for i in range(8)}
    st.table({"Register": list(reg_rows.keys()), "Value": list(reg_rows.values())})
    st.write(f"PC: {cpu.pc}   |   ZF: {cpu.zf}   |   HALTED: {cpu.halted}")

    st.subheader("Memory Inspector")
    start = st.number_input("Start address", min_value=0, max_value=65535, value=96)
    count = st.number_input("Count", min_value=1, max_value=256, value=32)
    end = min(65536, int(start)+int(count))
    mem_slice = cpu.memory[int(start):end]
    st.table({"Addr": list(range(int(start), end)), "Val": mem_slice})

st.caption("Supported ops: NOP, HALT, MOV, LOAD, STORE, ADD, SUB, CMP, JMP, JZ, JNZ, PUSH, POP, CALL, RET.  R7 is SP.")
