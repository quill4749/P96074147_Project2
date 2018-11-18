import numpy as np
import pandas as pd
import os
import sys
from sklearn import tree
from sklearn import preprocessing
import pydotplus
import collections
from time import gmtime, strftime

def conda_fix(graph):
        path = os.path.join(sys.base_exec_prefix, "Library", "bin", "graphviz")
        paths = ("dot", "twopi", "neato", "circo", "fdp")
        paths = {p: os.path.join(path, "{}.exe".format(p)) for p in paths}
        graph.set_graphviz_executables(paths)

os.chdir('C:\\Users\\Yeh Hsin-Yu\\Desktop\\P96074147_Project2')

dataFeature = ["on_time", "schedule", "meeting", "grade", "courses"]

trainData = pd.read_csv("data_train.csv")
testData = pd.read_csv("data_test.csv")

trainer = pd.DataFrame([trainData["on_time"], trainData["schedule"], trainData["meeting"], trainData["grade"], trainData["courses"]]).T
tree_model = tree.DecisionTreeClassifier()
tree_model.fit(X = trainer,y = trainData["pressure"])

tree_model.score(X = trainer, y = trainData["pressure"])

dot_data = tree.export_graphviz(tree_model, feature_names=dataFeature, out_file=None, filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('red', 'blue')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

conda_fix(graph)
OUT_PNG_NAME = str(strftime("%Y%m%d%H%H%M%S", gmtime()))+".png"
graph.write_png(OUT_PNG_NAME)

test_features = pd.DataFrame([testData["on_time"], testData["schedule"], testData["meeting"], testData["grade"], testData["courses"]]).T
test_preds = tree_model.predict(X=test_features)

reportData = pd.DataFrame(test_preds)
reportData.to_csv("result.csv", index=False)
