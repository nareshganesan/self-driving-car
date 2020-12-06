# self-driving-car

Self driving car nano degree projects

| Task        |          Name           |
| ----------- | :---------------------: |
| [Project 1] | [Find lane lines basic] |

[project 1]: https://github.com/nareshganesan/self-driving-car/tree/main/udacity-nd/P1_lane_lines
[find lane lines basic]: https://github.com/nareshganesan/self-driving-car/tree/main/udacity-nd/P1_lane_lines

### development setup

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
