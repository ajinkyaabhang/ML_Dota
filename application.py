from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
import flask_monitoringdashboard as dashboard
from predictFromModel import prediction
import json
import pickle
from file_operations import file_methods
from application_logging import logger

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

application = Flask(__name__)
app = application
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict_json", methods=['POST'])
@cross_origin()
def predictJsonRouteClient():
    try:
        if request.json['folderPath'] is not None:
            path = request.json['folderPath']

            pred_val = pred_validation(path) #object initialization

            pred_val.prediction_validation() #calling the prediction_validation function

            pred = prediction(path)  # object initialization

            # predicting for dataset present in database
            path, json_predictions = pred.predictionFromModel()
            return Response("Prediction File created at !!!" + str(path) + ' and few of the predictions are ' + str(json.loads(json_predictions)))
        else:
           print('Nothing Matched')

    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    try:
        path = request.form['Default_File_Predict']

        pred_val = pred_validation(path)  # object initialization

        pred_val.prediction_validation()  # calling the prediction_validation function

        pred = prediction(path)  # object initialization

        # predicting for dataset present in database
        path, json_predictions = pred.predictionFromModel()

        return render_template('results.html',prediction='Prediction has been saved at {} and few of the predictions are '.format(path) +' ' + str(json.loads(json_predictions)))

    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)

@app.route("/predict_new", methods=['POST'])
@cross_origin()
def predictNewRouteClient():
    try:
        is_C13 = request.form['C13']
        if (is_C13 == '1' or is_C13 == '-1'):
            C13 = 1
        else:
            C13 = 0
        is_H18 = request.form['H18']
        if (is_H18 == '1' or is_H18 == '-1'):
            H18 = 1
        else:
            H18 = 0
        is_F46 = request.form['F46']
        if (is_F46 == '1' or is_F46 == '-1'):
            F46 = 1
        else:
            F46 = 0
        is_F56 = request.form['F56']
        if (is_F56 == '1' or is_F56 == '-1'):
            F56 = 1
        else:
            F56 = 0
        is_G57 = request.form['G57']
        if (is_G57 == '1' or is_G57 == '-1'):
            G57 = 1
        else:
            G57 = 0
        is_I59 = request.form['I59']
        if (is_I59 == '1' or is_I59 == '-1'):
            I59 = 1
        else:
            I59 = 0
        is_A71 = request.form['A71']
        if (is_A71 == '1' or is_A71 == '-1'):
            A71 = 1
        else:
            A71 = 0
        is_F76 = request.form['F76']
        if (is_F76 == '1' or is_F76 == '-1'):
            F76 = 1
        else:
            F76 = 0
        is_A91 = request.form['A91']
        if (is_A91 == '1' or is_A91 == '-1'):
            A91 = 1
        else:
            A91 = 0
        is_E115 = request.form['E115']
        if (is_E115 == '1' or is_E115 == '-1'):
            E115 = 1
        else:
            E115 = 0

        filename = "models/KMeans/KMeans.sav"
        loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
        # predictions using the loaded model file
        clusters=loaded_model.predict([[H18, F76, F46, G57, C13, A71, E115, F56, I59, A91]])
        file_object = open("Prediction_Logs/Prediction_Log_single.txt", 'a+')
        log_writer = logger.App_Logger()
        file_loader = file_methods.File_Operation(file_object, log_writer)

        model_name = file_loader.find_correct_model_file(clusters[0])
        model = file_loader.load_model(model_name)
        result = model.predict([[H18, F76, F46, G57, C13, A71, E115, F56, I59, A91]])
        if (result[0] == 0):
            output = 'Win(1)'
        else:
            output = 'Lose(-1)'
        log_writer.log(file_object, 'End of Prediction')
        file_object.close()

        return render_template('results.html',prediction='According to prediction you will {} DOTA'.format(output))

    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)

@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():

    try:
        if request.json['folderPath'] is not None:
            path = request.json['folderPath']
            train_valObj = train_validation(path) #object initialization

            train_valObj.train_validation()#calling the training_validation function

            trainModelObj = trainModel() #object initialization
            trainModelObj.trainingModel() #training the model for the files in the table


    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")

#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    #host = '0.0.0.0'
    #port = 5000
    #httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    #httpd.serve_forever()
    app.run(debug=True)
