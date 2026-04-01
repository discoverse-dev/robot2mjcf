1. 使用sw导出urdf+stl（白模），将urdf文件中的mesh路径改成相对路径
2. 根据urdf合并body，生成完整的机器人模型
    ```shell
    python align_stp/merge_urdf.py /path/to/your/robot.urdf
    ```
    使用`-o`可以指定生成文件的目录，默认是`/path/to/your/robot.urdf/../merged_model`
3. 导出带外观的完整机器人模型，一个obj+mlt 或者 一个dae文件 (robot_id.obj)
4.（可选）若高精度obj模型与stl合并后的obj模型坐标系/尺度不一致，先用 `align_meshes.py` 对齐：
    ```shell
    # 单对对齐
    python align_stp/align_meshes.py \
        --ref /path/to/your/white_model.STL \
        --src /path/to/your/high_res.obj \
        --out /path/to/your/aligned.obj

    # 批量对齐 (按文件名匹配)
    python align_stp/align_meshes.py \
        --ref-dir /path/to/your/stl_meshes \
        --src-dir /path/to/your/obj_meshes \
        --out-dir /path/to/your/aligned_meshes \
        --ref-ext .STL --src-ext .obj
    ```
    - 自动检测并补偿 10x/100x 等尺度差异 (mm↔m 等)，无需手动指定
    - `--no-scale` 禁用自动尺度补偿
    - `--no-global` 跳过全局配准 (两模型已大致对齐时)
    - `--voxel-size` / `--max-icp-dist` 手动指定参数 (默认 0=自动)
4. ，下面的visual.obj和mapping.json都是第2步生成的
    ```shell
    python align_stp/assign_mesh_part.py /path/to/your/robot_id.obj -g /path/to/your/robot.urdf/../merged_model/visual.obj -m /path/to/your/robot.urdf/../merged_model/mapping.json
    ```
    使用`-o`可以指定生成文件的路径，默认是`/path/to/your/robot_id.obj/../meshes_aligned`
5. 修改urdf文件`/path/to/your/robot.urdf`，将其中的visual mesh路径修改成`/path/to/your/robot_id.obj/../meshes_aligned`中的mesh路径
6. 执行urdf到mjcf转换：
    ```shell
    robot2mjcf /path/to/your/new/robot.urdf -o /path/to/your/new/mjcf/robot.xml --no-convex-decompose
    # (不做凸分解)
    ```