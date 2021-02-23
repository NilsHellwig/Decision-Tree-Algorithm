package com.company;
import java.io.*;
import java.util.*;
import java.io.PrintStream;
import java.io.FileOutputStream;


public class Main {

    public static String trainingDataSetPath = "src/dataset/train.csv";
    public static String testDataSetPath = "src/dataset/test.csv";

    public static void main(String[] args) throws FileNotFoundException {
        // Lade Trainingsdaten und split in train/test
        ArrayList<HashMap<String, String>> fullData = CsvHelper.readFile(trainingDataSetPath);
        HashMap<String, ArrayList<HashMap<String, String>>> data = DecisionTreeClassifier.trainTestValidateSplit(fullData, 0.0,0);

        // Trainiere Classifier mit allen Trainingsdaten
        DecisionTreeClassifier dtc = getClassifier(data);

        // Erstelle valide Datei, die bei Kaggle hochgeladen werden kann, um Accuracy zu bestimmen
        evaluateClassifierTrainedOnAllTrainingExamples(dtc);

        /* Evaluiere Training, wobei aus test.csv zufällig 80% Training 20% Test Dokumente gewählt werden
           Dies wird fünf mal wiederholt, sodass ein Mittelwert von 5 Accuracies berechnet wird*/
        evaluate();
    }

    public static void evaluateClassifierTrainedOnAllTrainingExamples(DecisionTreeClassifier dtc) throws FileNotFoundException {
        try {
            ArrayList<HashMap<String, String>> preds = dtc.predictCsv(testDataSetPath);
            
            // write prediction_test.csv in format acceptable for kaggle
            ArrayList<HashMap<String, String>> outMaps = new ArrayList<HashMap<String, String>>();
            for(int i = 0; i<preds.size(); i++) {
                HashMap<String, String> outMap = new HashMap<String, String>();
                HashMap<String, String> pred = preds.get(i);
                outMap.put("PassengerId", pred.get("PassengerId"));
                outMap.put("Survived", pred.get("prediction"));
                outMaps.add(outMap);
            }

            CsvHelper.writeFile("./predictions_test.csv", outMaps, ',');
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static void evaluate() throws FileNotFoundException {
        double averagePrecision = 0;
        for(int i = 0; i<5; i++) {
            // Definiere Pfad der Traingsdaten
            String trainingDataSetPath = "src/dataset/train.csv";

            // Lade Trainingsdaten und split in train/test
            ArrayList<HashMap<String, String>> fullData = CsvHelper.readFile(trainingDataSetPath);
            HashMap<String, ArrayList<HashMap<String, String>>> data = DecisionTreeClassifier.trainTestValidateSplit(fullData, 0.2,0);

            // Train Classifier
            DecisionTreeClassifier decisionTreeClassifier = getClassifier(data);

            try {
                ArrayList<HashMap<String, String>> preds = decisionTreeClassifier.predictDataset(data.get("test"), "Survived");
                averagePrecision += decisionTreeClassifier.getLastPredictionPrecision();
                CsvHelper.writeFile("./predictions.csv", preds);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        averagePrecision /= 5;
        System.out.println(new StringBuilder().append("average precision over 5 iterations: ").append(averagePrecision));
    }

    public static DecisionTreeClassifier getClassifier(HashMap<String, ArrayList<HashMap<String, String>>> data) throws FileNotFoundException {
        // Definiere, welche Attribute genutzt werden sollen, um das TargetAttribute vorauszusagen
        List<String> attributesList = Arrays.asList("Pclass", "Sex", "SibSp", "Age", "Parch", "Embarked", "Fare");
        ArrayList<String> attributes = new ArrayList<>(attributesList);

        // Definiere, welches Attribut vorausgesagt werden soll
        String targetAttribute = "Survived";

        // Create new Classifier
        DecisionTreeClassifier dtc = new DecisionTreeClassifier(data.get("train"), attributes);

        // Erstelle Printstream um trainierten Graphen in .dot File zu speichern
        PrintStream logStream;
        logStream = new PrintStream(new FileOutputStream("decision_tree.dot"));
        logStream.println("digraph G{");
        dtc.setLogStream(logStream);

        // Discretize values
        dtc.registerDiscretization("Age", 8);
        dtc.registerDiscretization("Fare", 1);
        dtc.registerDiscretization("Parch", 6);

        // Trainiere den Baum
        dtc.trainDecisionTree(targetAttribute);

        // Ende der .dot File schreiben
        logStream.println("}");
        logStream.close();
        return dtc;
    }

}
