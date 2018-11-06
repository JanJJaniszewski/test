"""config file"""
from os.path import join as paths
import os

# ROOT
ROOT = os.path.dirname(os.path.abspath(__file__))

# ROOT:Data
data = paths(ROOT, 'Data')
input = paths(data, 'Input')
train = paths(input, 'train.csv')
test = paths(input, 'test.csv')

# ROOT:Data:Input
output = paths(ROOT, 'Input')

# ROOT:Data:Input:Analyses
analyses = paths(ROOT, 'Analyses')

# ROOT:Analyses:Model
model = paths(analyses, 'Some_topic')

# ROOT:Analyses:Model:model_results
subfolders = os.listdir(model)
subfolders.sort()
currentVersion = subfolders[-1]
model_results = paths(model, currentVersion, 'Results')
model_results_pdf = paths(model_results, 'Model_results.pdf')
tree = paths(model_results, 'Tree.png')

# ROOT:SideProjects
sideprojects = paths(analyses, 'Sideprojects')

# ROOT:Data:Input
input = paths(data, 'Input')

# ROOT:Data:Data
throughput = paths(data, 'Data', 'Throughput')

# ROOT:Functions
functions = paths(ROOT, 'Functions')

# ROOT:Logs
logs = paths(model, 'Logs')
model_log = paths(logs, 'Pipes.log')
gridsearch_log = paths(logs, 'gridsearchresults.csv')

if __name__ == '__main__':
    pass
