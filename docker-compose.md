To run the docker-compose for Isaac Sim 6.0.1

1. Pull the docker image:
   ```  
   docker pull nvcr.io/nvidia/isaac-sim:6.0.1
   ``` 
2. Run the docker-compose:
   ```  
   docker compose up -d
   ``` 
3. To access the terminal of the docker root container:
   ``` 
   docker exec -u 0 -it isaac-sim bash
   ``` 