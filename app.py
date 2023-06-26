from flask import Flask, render_template, request,jsonify, url_for
import base64
import threading
import json
import time

import sys
from os import path


import os
# Authentication package
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml.entities import Environment
from azure.ai.ml import MLClient

app = Flask(__name__, template_folder='template')
credential = DefaultAzureCredential()
#retourne True si le model existe sinon False
def test_model_exist(model_name):
    # Configuration de l'accès à AWS
    #test2

    try:
        # Vérifier si le modèle existe
        response = sagemaker_client.describe_model(ModelName=model_name)

        if 'ModelName' in response:
            # Supprimer le modèle
            value = True
            print(f"Le modèle {model_name} existe.")
        else:
            value = False
            print(f"Le modèle {model_name} n'existe pas.")
    except ClientError as e:
        value = False
        error_message = e.response['Error']['Message']
        if 'Could not find model' in error_message:
            print(f"Le modèle {model_name} n'existe pas.")
        else:
            print(f"Une erreur s'est produite lors de la suppression du modèle {model_name}: {error_message}")
    return value



#supprime le model s'il existe
def delete_sagemaker_model(model_name):
    # Configuration de l'accès à AWS


    try:
        # Vérifier si le modèle existe
        response = sagemaker_client.describe_model(ModelName=model_name)

        if 'ModelName' in response:
            # Supprimer le modèle
            sagemaker_client.delete_model(ModelName=model_name)
            print(f"Le modèle {model_name} a été supprimé avec succès.")
        else:
            print(f"Le modèle {model_name} n'existe pas.")
    except ClientError as e:
        error_message = e.response['Error']['Message']
        if 'Could not find model' in error_message:
            print(f"Le modèle {model_name} n'existe pas.")
        else:
            print(f"Une erreur s'est produite lors de la suppression du modèle {model_name}: {error_message}")


#supprime l'endpoint s'il existe
def delete_endpoint_from_sagemaker(endpoint_name):
    # Delete the endpoint configuration
    try:
        response = sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        # Supprimer l'endpoint
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print("Endpoint configuration deleted successfully.")
        res = "Endpoint configuration deleted successfully"
    except Exception as e:
        print("Error occurred while deleting the endpoint configuration:", str(e))
        res = "l'endpoint n'existe pas "
    return res

#retourne True si l'endpoint existe sinon False
def test_endpoint_existence(endpoint_name):

    try:
        # Appeler la méthode describe_endpoint pour vérifier si l'endpoint existe
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        print("L'endpoint existe.")
        return True
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "ValidationException" and "Could not find endpoint" in str(e):
            print("L'endpoint n'existe pas.")
            return False
        else:
            # Gérer d'autres exceptions ici si nécessaire
            raise e

def delete_model_file_from_s3(s3_bucket, s3_key):
    client = boto3.client('s3')
    response = client.list_objects(
        Bucket=s3_bucket,
        Prefix=s3_key
    )
    obj_list = []
    for data in response.get('Contents', []):
        print('res', data.get('Key'))
        obj_list.append({'Key': data.get('Key')})
    if obj_list:
        response = client.delete_objects(
            Bucket=s3_bucket,
            Delete={'Objects': obj_list}
        )
        print('response', response)


def deployendpoint():

    #get model train name from s3

    # Récupérer le contenu du fichier

    print("avant le get")
    # Créer une instance du client S3
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=s3_bucket, Key=file_name)
    content = response['Body'].read().decode('utf-8')
    print("content :", content)
    print("apres le get")
    # get the model
    model = sagemaker.Model(
        image_uri='778331702232.dkr.ecr.eu-west-3.amazonaws.com/nv-pred-sagemaker:latest',
        model_data='s3://nvabucket/output/'+content+'/output/model.tar.gz',
        role='arn:aws:iam::778331702232:role/nvsagemaker',
        sagemaker_session=sagemaker_session,
        name=model_name
    )

    # Deploy the model
    predictor = model.deploy(
        initial_instance_count=instance_count,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )

    # Get the endpoint
    predictor = sagemaker.predictor.Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_session
    )

#penser à appeler avant de lancer : getendpointormodelfrombucket avec comme parametre modelid/ et modelid.txt
@app.route("/deploy", methods=["POST"])
def deploy():
    print("avant try")
    try:
        model = test_model_exist(model_name)
        print("test model ?")
        if model:

            # vérifier s'il n'y a pas deja un endpoint avec ce model
            old_endpoint_id = test_endpoint_existence(endpoint_name)
            #si lendpoint existe déja
            if(old_endpoint_id):
                print(f"l'endpoint {endpoint_name} existe déjà avec l'id : ")
                res = "un endpoint existe déjà supprimez le avant d'en recréer un"
                #TODO : voir pour faire une redirection ou mettre a jour la page avec ce message
                return render_template('deploy.html', result=res)
            else:
                print("avant le thread du deploy")
                thread = threading.Thread(target=deployendpoint)
                thread.start()
                res = "l'endpoint a bien ete deploy"
                return render_template('deploy.html', result=res)
        else:
            res = "aucun model n'existe il faut faire un train"
            print(f"Le modèle avec l'ID {model_name} n'existe pas.")
            return render_template('deploy.html', result=res)

    except NotFound:
        # Le modèle n'existe pas
        print(f"Le modèle {model_name} n'existe pas.")
        res = "aucun model n'existe il faut faire un train"
        # TODO : voir pour faire une redirection ou mettre a jour la page avec ce message

    except Exception as e:
        # Gérez les exceptions possibles lors de l'interaction avec l'API AI Platform
        print(f"Une erreur s'est produite : {str(e)}")

    print("Fin de la vérification du modèle")

#penser à appeler avant de lancer : getendpointormodelfrombucket avec comme parametre endpoint/ et endpointid.txt
@app.route("/delete", methods=["POST"])
def delete():
    res = delete_endpoint_from_sagemaker(endpoint_name)
    return render_template('deploy.html', result=res)
@app.route("/trainfunc", methods=["POST"])
def trainfunc():
    # Get a handle to the workspace
    ml_client = MLClient(
        credential=credential,
        subscription_id="d9414828-c496-44ab-b6f9-e903bce40957",
        resource_group_name="mlops",
        workspace_name="nvml",
    )
    registered_model_name = 'credit_defaults_model'

    dependencies_dir = "./dependencies"
    os.makedirs(dependencies_dir, exist_ok=True)

    custom_env_name = "aml-scikit-learn"

    # Name assigned to the compute cluster
    cpu_compute_target = "cpu-cluster"

    try:
        # let's see if the compute target already exists
        cpu_cluster = ml_client.compute.get(cpu_compute_target)
        print(
            f"You already have a cluster named {cpu_compute_target}, we'll reuse it as is."
        )

    except Exception:
        print("Creating a new cpu compute target...")

        # Let's create the Azure ML compute object with the intended parameters
        cpu_cluster = AmlCompute(
            name=cpu_compute_target,
            # Azure ML Compute is the on-demand VM service
            type="amlcompute",
            # VM Family
            size="STANDARD_DS2_V2",
            # Minimum running nodes when there is no job running
            min_instances=0,
            # Nodes in cluster
            max_instances=4,
            # How many seconds will the node running after the job termination
            idle_time_before_scale_down=180,
            # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
            tier="Dedicated",
        )
        print(
            f"AMLCompute with name {cpu_cluster.name} will be created, with compute size {cpu_cluster.size}"
        )
        # Now, we pass the object to MLClient's create_or_update method
        cpu_cluster = ml_client.compute.begin_create_or_update(cpu_cluster)



    # TODO WAIT ICI

    pipeline_job_env = Environment(
        name=custom_env_name,
        description="Custom environment for Credit Card Defaults pipeline",
        tags={"scikit-learn": "0.24.2"},
        conda_file=os.path.join(dependencies_dir, "conda.yml"),
        image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
    )
    pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

    print(
        f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}"
    )


    train_src_dir = "./src"
    os.makedirs(train_src_dir, exist_ok=True)

    from azure.ai.ml import command
    from azure.ai.ml import Input

    job = command(
        code="./src/",  # location of source code
        command="python main.py",
        environment="aml-scikit-learn@latest",
        compute=cpu_compute_target,
        experiment_name="train_model_credit_default_prediction",
        display_name="credit_default_prediction",
    )

    ml_client.create_or_update(job)
    print("end")

def retrainfunc():

    timestamp = str(int(time.time()))  # Obtenir le timestamp actuel
    nom_nv = "nv" + timestamp
    s3 = boto3.client('s3')
    s3.put_object(Body=nom_nv, Bucket=s3_bucket, Key=file_name)
    print(nom_nv)

    estimator = sagemaker.estimator.Estimator(
        image_uri='778331702232.dkr.ecr.eu-west-3.amazonaws.com/nv-train-sagemaker:latest',
        role='arn:aws:iam::778331702232:role/nvsagemaker',
        instance_count=instance_count,
        instance_type=instance_type,
        output_path='s3://nvabucket/output',
        sagemaker_session=sagemaker_session,
        max_run=24 * 60 * 60
    )

    estimator.fit(job_name=nom_nv)

    #TODO : voir ou mettre car la ça va supprimer aussi ce que je viens de créer mais
    # avant j'ai besoin pour retrain donc peu être dans le code train
    delete_model_file_from_s3(s3_bucket, 'output')

    # supprimer l'ancien endpoint s'il existe
    delete_endpoint_from_sagemaker(endpoint_name)

    # supprimer l'ancien model s'il existe
    delete_sagemaker_model(model_name)
    modeldata = "s3://nvabucket/output/" + nom_nv + "/output/model.tar.gz"
    print("modeldata :", modeldata)
    # Create the model
    model = sagemaker.Model(
        image_uri='778331702232.dkr.ecr.eu-west-3.amazonaws.com/nv-pred-sagemaker:latest',
        model_data=modeldata,
        role='arn:aws:iam::778331702232:role/nvsagemaker',
        sagemaker_session=sagemaker_session,
        name='nvamodel'
    )

    # Deploy the model
    predictor = model.deploy(
        initial_instance_count=instance_count,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )

    # Get the endpoint
    predictor = sagemaker.predictor.Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_session
    )
    print("end")
    dest = 'retrain/'

    #supprimer l'ancien endpoint s'il existe
    delete_endpoint_from_sagemaker(endpoint_name)

    #supprimer l'ancien model s'il existe
    delete_sagemaker_model(model_name)

    print("before delete")
    bucket = s3.Bucket(s3_bucket)
    malignant_blobs = bucket.objects.filter(Prefix=dest + 'malignant/')
    for blob in malignant_blobs:
        blob.delete()

    # Supprimer les objets dans le préfixe benign
    benign_blobs = bucket.objects.filter(Prefix=dest + 'benign/')
    for blob in benign_blobs:
        blob.delete()
    print("after delete")

@app.route("/")
def template_test():

    return render_template('index.html', my_string="Wheeeee!", my_list=[0,1,2,3,4,5])



@app.route('/train', methods=['POST'])
def train():
    print("avant le thread du train")
    thread = threading.Thread(target=trainfunc)
    thread.start()
    return render_template('retrain.html', result="en cours")
    #return render_template('retrain.html', result=deployed_model_id)
    #return render_template('retrain.html', result="ok")
@app.route('/retrain', methods=['POST'])
def retrain():

    print("before send to bucket")
    print("malignant")
    malignant = request.files.getlist('malignant')

    print(malignant)
    benign = request.files.getlist('benign')
    print(benign)

    print("dans le retrain")
    print(malignant)
    # Ajouter les dossiers dans le bucket gcp
    s3 = boto3.resource('s3')
    dest = 'retrain/'

    # Ajouter les fichiers du dossier malignant
    for file in malignant:
        # Charger l'image dans S3
        s3.Object(s3_bucket,dest + file.filename).put(Body=file)

    print("malignant load ok")
    # Ajouter les fichiers du dossier benign
    for file in benign:
        s3.Object(s3_bucket,dest + file.filename).put(Body=file)

    thread = threading.Thread(target=retrainfunc)
    thread.start()
    return render_template('retrain.html', result="en cours")


@app.route('/upload', methods=['POST'])
def upload():

    # Get the endpoint
    predictor = sagemaker.predictor.Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_session
    )
    # Récupérer le fichier envoyé dans la requête POST
    file = request.files['file']
    print(file)
    print("avant")
    encoded_string = base64.b64encode(file.read()).decode('utf-8')

    payload = {
        "instances": [
            {
                "image_bytes": {
                    "b64": encoded_string
                }
            }
        ]
    }
    payload_json = json.dumps(payload)

    #donner les authorisation d'utiliser l'endpoint
    predictions = predictor.predict(payload_json, initial_args={"ContentType": "application/json"})

    prediction_str = predictions.decode('utf-8')  # Decode byte string to Unicode string

    prediction_json = json.loads(prediction_str)  # Parse the string as JSON

    predictions = prediction_json['predictions']
    resultat = prediction_json['resultat']


    return render_template('result.html', result=resultat)

if __name__ == "__main__":
    app.run(debug=True)
