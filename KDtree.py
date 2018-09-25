"""
此文件为kd树的代码实现
算法描述：平衡kd树

1.构建kd树
设数据集为k维x向量
首先以x1的中位数为界，将N个数据分为两半
而后以x2的中位数为界……依此递归直到每个叶节点只包含一个数据点

2.查找最近邻
首先将目标点的坐标依次与节点对比，往下搜索直到叶节点
而后向上搜索，判断其他节点的超矩形是否与当前最近点与目标点组成的超球体相交
若不相交，则该节点的所有子节点可全部跳过
若相交，则检查该节点的子节点的超矩形

当搜索完根节点时，搜索完成

注意：每一个节点的超矩形范围由其父节点与二级父节点构成（父节点不存在时）
"""
