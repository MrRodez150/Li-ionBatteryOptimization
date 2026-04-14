sudo docker build -t liion-opt .

sudo docker run -it --name liion_run -v /home/Rodez/Li-ionBatteryOptimization:/workspace liion-opt bash

python main_run_parallel.py




sudo docker ps -a

sudo docker attach liion_run

sudo docker rm -f liion_run