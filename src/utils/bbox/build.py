import requests
import subprocess
files = ['nms.pyx', 'bbox.pyx', 'make.sh']
for file in files:
    res = requests.get(
        url='https://raw.githubusercontent.com/Mohamed209/CTPN-text-detector/master/src/utils/bbox/'+file)
    with open(file, mode='w') as code:
        code.writelines(res.text)

subprocess.run(['chmod', '+x', 'make.sh'])
subprocess.run(['bash', 'make.sh'])
