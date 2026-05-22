# 两台机器协作 Git 流程

目标：

- `task3` 作为稳定主分支。
- 电脑只在 `pc-task3` 上开发。
- Xavier 只在 `xavier-task3` 上开发。
- 最后再把两个工作分支合并回 `task3`。

## 第一次创建分支

电脑：

```bash
git checkout task3
git pull --rebase origin task3
git checkout -b pc-task3
git push -u origin pc-task3
```

Xavier：

```bash
git checkout task3
git pull --rebase origin task3
git checkout -b xavier-task3
git push -u origin xavier-task3
```

## 每次开始工作前

电脑：

```bash
git checkout pc-task3
git fetch origin
git rebase origin/task3
git pull --rebase origin pc-task3
```

Xavier：

```bash
git checkout xavier-task3
git fetch origin
git rebase origin/task3
git pull --rebase origin xavier-task3
```

如果提示有未提交修改：

```bash
git stash push -u -m "wip before sync"
git fetch origin
git rebase origin/task3
git pull --rebase origin 当前分支名
git stash pop
```

## 修改完成后提交到自己的分支

```bash
git status
git add .
git commit -m "描述这次修改"
git fetch origin
git rebase origin/task3
git push --force-with-lease
```

说明：

- 对自己的工作分支可以用 `--force-with-lease`，因为 rebase 会改提交历史。
- 不要对 `task3` 使用 `--force` 或 `--force-with-lease`。

## 合并回 task3

在一台机器上统一合并：

```bash
git checkout task3
git fetch origin
git pull --rebase origin task3

git merge origin/pc-task3
git merge origin/xavier-task3

git push origin task3
```

如果有冲突，打开冲突文件，处理这些标记：

```text
<<<<<<< HEAD
当前 task3 内容
=======
要合并进来的分支内容
>>>>>>> 分支名
```

处理完：

```bash
git add 冲突文件
git commit
git push origin task3
```

## 推荐分工

电脑主要改代码：

```text
grasp_tag_demo.py
monitor_servos.py
three_Inverse_kinematics.py
reset_arm.py
```

Xavier 主要改现场参数：

```text
calibration.json
```

尽量不要两台机器同时改同一个文件，尤其是：

```text
calibration.json
grasp_tag_demo.py
three_Inverse_kinematics.py
```

## 常见问题

### cannot pull with rebase: You have unstaged changes

意思是本地有未提交修改。先提交或 stash：

```bash
git stash push -u -m "wip before pull"
git pull --rebase origin 当前分支名
git stash pop
```

### push 被 rejected: non-fast-forward

说明远端比本地新。对自己的工作分支：

```bash
git fetch origin
git rebase origin/task3
git push --force-with-lease
```

对 `task3` 不要强推，先 pull/rebase 或处理 merge。

### 不想提交二进制和缓存

确认 `.gitignore` 已经包含：

```text
__pycache__/
Log/
*.so
*.engine
```

如果 `.so` 已经被 Git 跟踪过，需要取消跟踪：

```bash
git rm --cached libs/libmyplugins.so libs/yolov5_trt_cpp.so
git commit -m "stop tracking generated shared libraries"
```
