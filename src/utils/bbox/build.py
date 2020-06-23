import subprocess
files = ['nms.pyx', 'bbox.pyx', 'make.sh']
for file in files:
    subprocess.run(
        'wget --no-check-certificate --content-disposition https://raw.githubusercontent.com/Mohamed209/CTPN-text-detector/master/src/utils/bbox/'+file)
subprocess.run('chmod +x make.sh')
subprocess.run('bash make.sh')
