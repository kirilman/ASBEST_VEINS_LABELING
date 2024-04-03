
<div align="center">
  <p>
    <a align="center" href="">
      <img
        width="100%"
        src="./images/main.jpg"
      >
    </a>
  </p>
</div>

# ASBEST_VEINS_LABELING Test of utilits https://colab.research.google.com/drive/1rdzuZu-IdGFM29TakmpaGpIcF3FnuttW?usp=sharing
## Установка 
pip install -e .

## Запуск СVAT
1. docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
Запуск nuctl function для Sam
2. ./deploy_cpu.sh pytorch/facebookresearch/sam/

Ошибка запуска 
docker compose -f docker-compose.yml -f docker-compose.dev.yml -f components/serverless/docker-compose.serverless.yml up -d --build