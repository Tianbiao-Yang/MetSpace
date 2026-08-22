from pathlib import Path
import re

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


# ============================================================
# 输入文件
# ============================================================
INPUT_FILE = Path("../data/BA_prioritized_candidates_set.txt")


# ============================================================
# 输出目录
# ============================================================
OUT_DIR = Path("../result/chemical_structures")

OUT_PNG = OUT_DIR / "png"
OUT_2D_SDF = OUT_DIR / "2D_sdf"
OUT_3D_SDF = OUT_DIR / "3D_sdf"

for d in [OUT_DIR, OUT_PNG, OUT_2D_SDF, OUT_3D_SDF]:
    d.mkdir(parents=True, exist_ok=True)

N_CONFS = 1
RANDOM_SEED = 2025


# ============================================================
# 文件名安全处理
# ============================================================
def safe_filename(name):
    """
    防止 MS_ID 中有特殊字符导致文件名出错。
    """
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


# ============================================================
# 读取输入文件
# ============================================================
def read_smiles_file(input_file):
    """
    读取格式类似：

    MS_ID Smiles
    MS0001 CCCCCNC(=O)...
    MS0002 C[C@H](CCC(=O)O)...

    支持空格或 tab 分隔。
    """
    records = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = line.split(maxsplit=1)

            if len(parts) < 2:
                print(f"跳过第 {line_number} 行，无法解析: {line}")
                continue

            ms_id = parts[0].strip()
            smiles = parts[1].strip()

            # 跳过表头
            if ms_id.lower() in ["ms_id", "id"] and smiles.lower() in ["smiles", "smile"]:
                continue

            records.append((ms_id, smiles))

    return records


# ============================================================
# 生成 2D 分子
# ============================================================
def make_2d_mol(mol):
    """
    生成带 2D 坐标的分子。
    2D SDF 中 z 坐标通常为 0。
    """
    mol2d = Chem.Mol(mol)

    Chem.AssignStereochemistry(
        mol2d,
        force=True,
        cleanIt=True
    )

    rdDepictor.Compute2DCoords(mol2d)

    if mol2d.GetNumConformers() > 0:
        conf = mol2d.GetConformer()
        conf.Set3D(False)

        # 在 2D 结构中尽量显示楔形键/虚线键来表示手性
        Chem.WedgeMolBonds(mol2d, conf)

    return mol2d


# ============================================================
# 保存 PNG 二维结构图
# ============================================================
def save_png(mol2d, png_path, legend=""):
    """
    保存二维结构 PNG 图片。
    """
    width = 700
    height = 500

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)

    options = drawer.drawOptions()
    options.addStereoAnnotation = True
    options.legendFontSize = 24

    drawer.DrawMolecule(mol2d, legend=legend)
    drawer.FinishDrawing()

    png_path.write_bytes(drawer.GetDrawingText())


# ============================================================
# 生成 3D 分子
# ============================================================
def make_3d_mol(mol, n_confs=1, random_seed=2025):

    mol3d = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.enforceChirality = True
    params.pruneRmsThresh = 0.5

    # 不同 RDKit 版本可能属性略有差异，因此用 hasattr 判断
    if hasattr(params, "useSmallRingTorsions"):
        params.useSmallRingTorsions = True

    if hasattr(params, "useMacrocycleTorsions"):
        params.useMacrocycleTorsions = True

    conf_ids = list(
        AllChem.EmbedMultipleConfs(
            mol3d,
            numConfs=n_confs,
            params=params
        )
    )

    # 如果常规嵌入失败，使用随机坐标再尝试一次
    if len(conf_ids) == 0:
        params.useRandomCoords = True

        conf_ids = list(
            AllChem.EmbedMultipleConfs(
                mol3d,
                numConfs=n_confs,
                params=params
            )
        )

    if len(conf_ids) == 0:
        raise RuntimeError("3D conformer embedding failed")

    # 优先使用 MMFF94s
    if AllChem.MMFFHasAllMoleculeParams(mol3d):
        opt_results = AllChem.MMFFOptimizeMoleculeConfs(
            mol3d,
            numThreads=0,
            maxIters=1000,
            mmffVariant="MMFF94s"
        )
        force_field = "MMFF94s"
    else:
        opt_results = AllChem.UFFOptimizeMoleculeConfs(
            mol3d,
            numThreads=0,
            maxIters=1000
        )
        force_field = "UFF"

    # opt_results 格式为：
    # [(not_converged, energy), ...]
    energies = [result[1] for result in opt_results]

    best_idx = min(range(len(energies)), key=lambda i: energies[i])
    best_conf_id = conf_ids[best_idx]
    best_energy = energies[best_idx]
    not_converged = opt_results[best_idx][0]

    # 只保留能量最低的构象
    best_conf = Chem.Conformer(mol3d.GetConformer(best_conf_id))
    best_conf.Set3D(True)

    final_mol = Chem.Mol(mol3d)
    final_mol.RemoveAllConformers()
    final_mol.AddConformer(best_conf, assignId=True)

    return final_mol, best_energy, force_field, not_converged


# ============================================================
# 保存单个 SDF
# ============================================================
def save_single_sdf(mol, sdf_path):
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()


# ============================================================
# 主程序
# ============================================================
def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到输入文件: {INPUT_FILE}")

    records = read_smiles_file(INPUT_FILE)

    print(f"读取到 {len(records)} 个分子")

    all_2d_sdf_path = OUT_DIR / "all_2D.sdf"
    all_3d_sdf_path = OUT_DIR / "all_3D.sdf"
    failed_path = OUT_DIR / "failed.txt"
    summary_path = OUT_DIR / "summary.tsv"

    writer_all_2d = Chem.SDWriter(str(all_2d_sdf_path))
    writer_all_3d = Chem.SDWriter(str(all_3d_sdf_path))

    failed_records = []
    summary_records = []

    n_success_2d = 0
    n_success_3d = 0

    for idx, (ms_id, smiles) in enumerate(records, start=1):
        print(f"[{idx}/{len(records)}] Processing {ms_id}")

        file_id = safe_filename(ms_id)

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            reason = "SMILES parse failed"
            print(f"  失败: {reason}")
            failed_records.append((ms_id, smiles, reason))
            summary_records.append((ms_id, smiles, "failed", "failed", "", "", reason))
            continue

        # 保存基础属性
        mol.SetProp("_Name", ms_id)
        mol.SetProp("MS_ID", ms_id)
        mol.SetProp("SMILES", smiles)

        two_d_status = "failed"
        three_d_status = "failed"
        energy_text = ""
        force_field_text = ""
        reason_text = ""

        # ====================================================
        # 生成 PNG 和 2D SDF
        # ====================================================
        try:
            mol2d = make_2d_mol(mol)

            mol2d.SetProp("_Name", ms_id)
            mol2d.SetProp("MS_ID", ms_id)
            mol2d.SetProp("SMILES", smiles)
            mol2d.SetProp("Structure_dimension", "2D")

            png_path = OUT_PNG / f"{file_id}.png"
            sdf2d_path = OUT_2D_SDF / f"{file_id}_2D.sdf"

            save_png(
                mol2d,
                png_path,
                legend=ms_id
            )

            save_single_sdf(
                mol2d,
                sdf2d_path
            )

            writer_all_2d.write(mol2d)

            n_success_2d += 1
            two_d_status = "success"

        except Exception as e:
            reason = f"2D generation failed: {e}"
            print(f"  2D 失败: {reason}")
            failed_records.append((ms_id, smiles, reason))
            reason_text += reason

        # ====================================================
        # 生成 3D SDF
        # ====================================================
        try:
            mol3d, best_energy, force_field, not_converged = make_3d_mol(
                mol,
                n_confs=N_CONFS,
                random_seed=RANDOM_SEED
            )

            mol3d.SetProp("_Name", ms_id)
            mol3d.SetProp("MS_ID", ms_id)
            mol3d.SetProp("SMILES", smiles)
            mol3d.SetProp("Structure_dimension", "3D")
            mol3d.SetProp("Force_field", force_field)
            mol3d.SetProp("Energy", str(best_energy))
            mol3d.SetProp("Optimization_not_converged", str(not_converged))

            sdf3d_path = OUT_3D_SDF / f"{file_id}_3D.sdf"

            save_single_sdf(
                mol3d,
                sdf3d_path
            )

            writer_all_3d.write(mol3d)

            n_success_3d += 1
            three_d_status = "success"
            energy_text = str(best_energy)
            force_field_text = force_field

        except Exception as e:
            reason = f"3D generation failed: {e}"
            print(f"  3D 失败: {reason}")
            failed_records.append((ms_id, smiles, reason))

            if reason_text:
                reason_text += "; " + reason
            else:
                reason_text = reason

        summary_records.append(
            (
                ms_id,
                smiles,
                two_d_status,
                three_d_status,
                force_field_text,
                energy_text,
                reason_text
            )
        )

    writer_all_2d.close()
    writer_all_3d.close()

    # ========================================================
    # 写入失败记录
    # ========================================================
    with open(failed_path, "w", encoding="utf-8") as f:
        f.write("MS_ID\tSMILES\tReason\n")
        for ms_id, smiles, reason in failed_records:
            f.write(f"{ms_id}\t{smiles}\t{reason}\n")

    # ========================================================
    # 写入 summary
    # ========================================================
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(
            "MS_ID\tSMILES\t2D_status\t3D_status\tForce_field\tEnergy\tReason\n"
        )

        for record in summary_records:
            f.write("\t".join(record) + "\n")

    print("\n完成！")
    print(f"总分子数: {len(records)}")
    print(f"成功生成 PNG/2D SDF: {n_success_2d}")
    print(f"成功生成 3D SDF: {n_success_3d}")
    print(f"失败记录数: {len(failed_records)}")
    print()
    print(f"PNG 输出目录: {OUT_PNG}")
    print(f"2D SDF 输出目录: {OUT_2D_SDF}")
    print(f"3D SDF 输出目录: {OUT_3D_SDF}")
    print(f"汇总 2D SDF: {all_2d_sdf_path}")
    print(f"汇总 3D SDF: {all_3d_sdf_path}")
    print(f"失败记录: {failed_path}")
    print(f"汇总表: {summary_path}")


if __name__ == "__main__":
    main()