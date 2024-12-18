from graphviz import Digraph

# 定义一个有横向布局的Digraph
dot = Digraph(comment='CAE Model Structure', graph_attr={'rankdir': 'LR', 'fontname': 'Arial'})

# 输入层
dot.node('Input', 'Input Layer\n28x28x1', shape='box')

# 编码器部分
dot.node('Conv1', 'Conv2D\n(3x3, 32)', shape='box')
dot.node('Pool1', 'MaxPooling2D\n(2x2)', shape='box')
dot.node('Conv2', 'Conv2D\n(3x3, 64)', shape='box')
dot.node('Pool2', 'MaxPooling2D\n(2x2)', shape='box')
dot.node('Bottleneck', 'Bottleneck\n(7x7x64)', shape='ellipse')

# 解码器部分
dot.node('Up1', 'UpSampling2D\n(2x2)', shape='box')
dot.node('Conv3', 'Conv2D\n(3x3, 32)', shape='box')
dot.node('Up2', 'UpSampling2D\n(2x2)', shape='box')
dot.node('Output', 'Output Layer\n28x28x1', shape='box')

# 数据流向
dot.edges([('Input', 'Conv1'), ('Conv1', 'Pool1'), ('Pool1', 'Conv2'), 
           ('Conv2', 'Pool2'), ('Pool2', 'Bottleneck'), ('Bottleneck', 'Up1'), 
           ('Up1', 'Conv3'), ('Conv3', 'Up2'), ('Up2', 'Output')])

# 保存图像
dot.render('cae_structure_horizontal', format='svg', cleanup=True)
dot.view()
