---
{"dg-publish":true,"permalink":"/tools/git-and-git-hub/","dgPassFrontmatter":true,"created":"2025-12-15T18:05:19.452+08:00","updated":"2026-01-06T20:55:34.744+08:00"}
---


# 从项目构建开始传入仓库

```bash
git add.

git commit -m "first commit"

git branch -M main

git remote add origin git@github.com:WaterDrop-EarthDivision/Personal-Blog.git

git push -u origin main
```

# 第二次及之后传入仓库

```bash

git pull origin main  # 获取最新

git status  # 查看修改

git add . 
git commit -m "你的提交信息" 
git push

```

# 克隆仓库

```bash
# 克隆仓库（自动关联 origin 并创建 main 分支）
git clone git@github.com:WaterDrop-EarthDivision/Personal-Blog.git
cd Personal-Blog

# 修改文件后...
git add .
git commit -m "docs: 更新博客文章"
git push
```

# 用法
## 0.1 完整流程

[https://blog.csdn.net/s_y_w123/article/details/112465793](https://blog.csdn.net/s_y_w123/article/details/112465793)

1. 先 git pull  
2. git add.  
3. git commit -m "msg"  
4. git push

## 1.将库中的代码推送到本地库中

```
git clone 远程地址
```

## 2.将自己代码推送至仓库中

```
git push 别名 本地库分支
```
## 3.将自己代码更新

```
git pull 别名 远程分支
```

## 4.初始化本地库

```
git init 初始化本地库
```

## 5.查看本地库状态

```
git status 查看本地库状态
```

## 6.查看本地库状态

```
git status 查看本地库状态
```

## 7.添加到暂存区

```
git add 文件名（全部为.）
```

## 8.提交到本地库

```
git commit -m "日志信息" 文件名 提交到本地库
```

## 9.查看历史记录

```
git reflog //版本信息 精简  
git log //查看版本详细信息
```

## 10.版本穿梭

```
git reset --hard 版本号
```
## 11.创建分支
```
git branch 分支名
```

## 12.查看分支

```
git branch -v 
```

## 13.切换分支

```
git checkout 分支名
```

## 14.把指定分支合并到当前分支上

```
git merge 分支名
```

## 15.查看当前所有远程地址别名

```
git remote -v
```

## 16.给远程地址添加别名

```
git remote add 别名 远程地址
```