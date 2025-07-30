#!/usr/bin/env python3
# eye_void_structured_asym.py —— 正方形域 + “不对称手绘眼睛”形空洞
#
# 依赖： gmsh ≥ 4.11      pip install gmsh
#
# -------------------------------------------------------------------------
# 运行：
#     python eye_void_structured_asym.py
#     gmsh meshes/eye_void_structured_asym.msh
# -------------------------------------------------------------------------

import math, gmsh, sys, os

# ================= 网格总体参数 ============================================
a   = 0.2          # 正方形单元边长 (m)
Nx  = 100           # 方板水平方向单元数   ⇒ 总宽 = Nx · a
Ny  = 100           # 方板垂直方向单元数   ⇒ 总高 = Ny · a

# ================= 眼睛外轮廓控制点 =======================================
# —— 上睑 ——   左眼角   →   最高点   →  右眼角
x_L_up,  y_L_up   = -0.2075 * Nx * a,  -0.065 * Ny * a            # 左角
x_P_up,  y_P_up   = 0 * Nx * a,  0.09 * Ny * a   # 上睑最高点（略偏右）
x_R_up,  y_R_up   =  0.2075 * Nx * a,  0.1 * Ny * a           # 右角
# 对各端点再给出斜率 dy/dx  (眼角尖锐程度、最高点处水平)
slope_L_up  =  2     # 往右上翘
slope_P_up  =  0     # 顶点水平
slope_R_up  =  0.85     # 往右下收

# —— 下睑 ——   左眼角   →   最低点   →  右眼角
x_L_lo,  y_L_lo   = x_L_up, y_L_up                    # 左角共用
x_P_lo,  y_P_lo   =  -0.0125 * Nx * a, -0.05 * Ny * a  # 下睑最低点（略偏右）
x_R_lo,  y_R_lo   = x_R_up, y_R_up                    # 右角共用
slope_L_lo  =  0.75    # 轻微下斜
slope_P_lo  =  0.05
slope_R_lo  =  0.5     # 轻微上扬（抬尾）

# ================= Hermite 三次多项式工具 ================================
def hermite(x, x1, x2, y1, y2, dy1, dy2):
    """端点为 (x1,y1),(x2,y2)，斜率 dy1,dy2 的三次 Hermite 插值"""
    t  = (x - x1) / (x2 - x1)
    h00 =  2*t**3 - 3*t**2 + 1
    h10 =      t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 =      t**3 -    t**2
    return (h00*y1 + h10*(x2-x1)*dy1 +
            h01*y2 + h11*(x2-x1)*dy2)

# ================= 上 / 下睑函数 y = f(x) =================================
def y_upper(x):
    if x < x_P_up:                                  # 左段
        return hermite(x,
                       x_L_up, x_P_up,
                       y_L_up, y_P_up,
                       slope_L_up, slope_P_up)
    else:                                           # 右段
        return hermite(x,
                       x_P_up, x_R_up,
                       y_P_up, y_R_up,
                       slope_P_up, slope_R_up)

def y_lower(x):
    if x < x_P_lo:
        return hermite(x,
                       x_L_lo, x_P_lo,
                       y_L_lo, y_P_lo,
                       slope_L_lo, slope_P_lo)
    else:
        return hermite(x,
                       x_P_lo, x_R_lo,
                       y_P_lo, y_R_lo,
                       slope_P_lo, slope_R_lo)

def in_void(px, py):
    """判断 (px,py) 是否落在眼睛空洞内"""
    # 超出左右眼角之外 → 肯定不在眼睛里
    if px < x_L_up or px > x_R_up:
        return False
    return (py <= y_upper(px)) and (py >= y_lower(px))

# ================== gmsh 初始化、顶点 ======================================
gmsh.initialize(sys.argv)
gmsh.model.add("eye_void_structured_asym")
gmsh.option.setNumber("Mesh.ElementOrder", 1)        # 一阶 ⇒ 不显示对角线

pt = {}                       # (row,col) → point tag
x0 = -0.5 * Nx * a
y0 = -0.5 * Ny * a

for row in range(Ny + 1):
    y = y0 + row * a
    for col in range(Nx + 1):
        x = x0 + col * a
        pt[(row, col)] = gmsh.model.geo.addPoint(x, y, 0, a)

# ================== 建面（跳过 void） =====================================
all_surf_tags = []
used_points   = set()

for row in range(Ny):
    for col in range(Nx):
        cx = x0 + (col + 0.5) * a
        cy = y0 + (row + 0.5) * a
        if in_void(cx, cy):                       # 落在空洞 → 跳过
            continue

        p1 = pt[(row,     col    )]
        p2 = pt[(row,     col + 1)]
        p3 = pt[(row + 1, col + 1)]
        p4 = pt[(row + 1, col    )]

        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)

        for l in (l1, l2, l3, l4):
            gmsh.model.geo.mesh.setTransfiniteCurve(l, 2)

        cl   = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        surf = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.mesh.setTransfiniteSurface(surf)
        gmsh.model.geo.mesh.setRecombine(2, surf)

        all_surf_tags.append(surf)
        used_points.update((p1, p2, p3, p4))

# ================== 删除孤立点 ============================================
for tag in pt.values():
    if tag not in used_points:
        gmsh.model.geo.remove([(0, tag)])

# ================== 生成网格 & 写文件 =====================================
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)

pg = gmsh.model.addPhysicalGroup(2, all_surf_tags)
gmsh.model.setPhysicalName(2, pg, "SquareWithAsymEyeVoid")

out_dir = "./meshes"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "eye_void_structured_asym.msh")

print(f"Writing '{out_file}' with {len(all_surf_tags)} square elements …")
gmsh.write(out_file)
gmsh.finalize()
print("Done.")
