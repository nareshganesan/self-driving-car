# self-driving-car

Self driving car nano degree projects

| Task        |            Name            |
| ----------- | :------------------------: |
| [Project 1] |  [Find lane lines basic]   |
| [Project 2] | [Find lane lines advanced] |

[project 1]: https://github.com/nareshganesan/self-driving-car/tree/main/udacity-nd/p1-finding-lane-lines
[find lane lines basic]: https://github.com/nareshganesan/self-driving-car/tree/main/udacity-nd/p1-finding-lane-lines
[project 2]: https://github.com/nareshganesan/self-driving-car/tree/main/udacity-nd/p2-advanced-lane-lines
[find lane lines advanced]: https://github.com/nareshganesan/self-driving-car/tree/main/udacity-nd/p2-advanced-lane-lines

### development host setup

```bash
# install pyenv
git clone https://github.com/pyenv/pyenv.git ~/.pyenv

# add bash exports
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# install python 3.7.2
pyenv install 3.7.2

# install self driving car virtualenv
pyenv virtualenv udSelfDriving

# activate environment
pyenv activate udSelfDriving

# dependencies NVIDIA driver, CUDA 10.1
# python pip virtualenv
pip install -r requirements.txt

# Jupyter lab extensions
jupyter labextension install @aquirdturtle/collapsible_headings
jupyter labextension install @lckr/jupyterlab_variableinspector
```

### development docker setup

```bash
# give docker group access to x server for display
xhost local:docker

# create docker with necessary dependencies
docker build -t aisdclab -f Dockerfile .

# run the docker with host display access
docker run --runtime=nvidia --rm \
 -p 8888:8888 \
 -e DISPLAY=\$DISPLAY \
 -v /tmp/.X11-unix/:/tmp/.X11-unix \
 -v `pwd`:/src \
 nareshganesan/aisdclab:latest /bin/bash
```
