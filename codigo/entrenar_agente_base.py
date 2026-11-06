import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import subprocess
import sys

max_iters = 1000 # Total number of training episodes
episode_iters = 20 # How many training episodes to execute before the training script is called again

# NOTICE: The training script trains and stores the models every 10 episode_iters. When the batch of episode_iters
# finishes, the model is stored and in the next itreation is loaded again to start the training were it was left before.
# This is to avoid some memory leak issue existing with TF

'''
Este archivo es el que permite entrenar el agente DLR, realizando un total de 'max_iters' epocas. En este caso, se entrena usando
lotes de épocas de tamaño 'episode_iters' por tema memoria

'''


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Master Training Script')
    parser.add_argument('-lr', type=float, default=0.0005)
    parser.add_argument('-T', type=int, default=8)
    parser.add_argument('-dim', type=int, default=32)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-readout_layers', type=int, default=4)
    parser.add_argument('-gamma', type=float, default=0.95)
    parser.add_argument('-vf_coef', type=float, default=0.5)
    parser.add_argument('-start_iter', type=int, default=0, help='Episodio desde el cual retomar el entrenamiento (0 por defecto)')
    parser.add_argument('-start_c', type=int, default=1, help='Contador del checkpoint desde el cual comenzar (1 por defecto)')
    parser.add_argument('-episode_iters', type=int, default=20, help='Episodios por subproceso')
    parser.add_argument('-max_iters', type=int, default=1000, help='Número máximo de episodios en los que parará la ejecución')
    args = parser.parse_args()

    max_iters = args.max_iters
    episode_iters = args.episode_iters

    if not os.path.exists("../logs"):
        os.makedirs("../logs")

    
    #Directorio temporal creado al comenzar la ejecución de este archivo, y justo antes de comenzar a entrenar el modelo.
    if not os.path.exists("../modelos/estado_entrenamiento"):
        os.makedirs("../modelos/estado_entrenamiento")

    iters = args.start_iter
    counter_store_model = args.start_c
    #Nombre de las topologías usadas en el proceso de entrenamiento (NO añadir NEW_ aquí, ya lo hace el sub-script)
    dataset_folder_name1 = "BtAsiaPac"
    dataset_folder_name2 = "Garr199905"
    dataset_folder_name3 = "Goodnet"

    while iters < max_iters:
        '''
        Pasamos como argumentos:
            * iters --> nuevo episodio por el que vamos en el entrenamiento
            * counter_store_model --> la nueva versión del modelo que se va a obtener
            * episode_iters ---> tamaño de lote de episodios (cuántos episodios procesar en el script train_Enero_3top_script.py)
        '''
        
        #subprocess.call(['python train_Enero_3top_script.py -i '+str(iters)+ ' -c '+str(counter_store_model)+' -e '+str(episode_iters)+ ' -f1 '+dataset_folder_name1+' -f2 '+dataset_folder_name2+' -f3 '+dataset_folder_name3], shell=True)
        #cmd = f"python train_Enero_3top_script_compresion.py -i {iters} -c {counter_store_model} -e {episode_iters} -f1 {dataset_folder_name1} -f2 {dataset_folder_name2} -f3 {dataset_folder_name3}"
        cmd = f"\"{sys.executable}\" entrenar_tanda_ppo.py -i {iters} -c {counter_store_model} -e {episode_iters} -f1 {dataset_folder_name1} -f2 {dataset_folder_name2} -f3 {dataset_folder_name3} -lr {args.lr} -T {args.T} -dim {args.dim} -dropout {args.dropout} -readout_layers {args.readout_layers} -gamma {args.gamma} -vf_coef {args.vf_coef}"
        
        try:
            subprocess.check_call(cmd, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR FATAL] Fallo en el sub-proceso durante procesado de episodio {iters}.")
            print(f"El proceso hijo finalizó con código de error: {e.returncode}")
            print("Abortando el bucle maestro del entrenamiento DRL para evitar iteraciones en vacío y sobreescritura errónea de modelos.")
            break
        
        counter_store_model = counter_store_model + episode_iters
        iters = iters + episode_iters






