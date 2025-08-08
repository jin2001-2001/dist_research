schdules.py _load_actions and load_csv -> _validate_and_set_stage_mapping -> _validate_schedule
                            |-> stage_index_to_group_rank (schedules.py and stage.py)

stage.py _shape_inference()

# 先卸载有冲突的包（在 qwencpu 环境里）
pip uninstall -y pyarrow pandas numpy datasets

# 安装彼此兼容、对 Pi 友好的版本
pip install "numpy==1.26.4" "pandas==2.1.4" "pyarrow==12.0.1" "datasets==2.14.6"
