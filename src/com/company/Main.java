package com.company;
import java.io.*;
import java.util.*;
import java.io.PrintStream;
import java.io.FileOutputStream;

public class Main {

    public static void main(String[] args) throws FileNotFoundException {
        // Definiere Pfad der Traingsdaten
        String trainingDataSetPath = "src/dataset/train.csv";

        //Lade Trainingsdaten und split in train/test
        ArrayList<HashMap<String, String>> fullData = CsvHelper.readFile(trainingDataSetPath);
        HashMap<String, ArrayList<HashMap<String, String>>> data = DecisionTreeClassifier.trainTestValidateSplit(fullData, 0.2, 0);

        // Definiere, welche Attribute genutzt werden sollen, um das TargetAttribute vorauszusagen
        List<String> attributesList = Arrays.asList("Pclass", "Sex", "SibSp", "Age", "Parch", "Embarked", "Fare");
        ArrayList<String> attributes = new ArrayList<>(attributesList);

        // Definiere, welches Attribut vorausgesagt werden soll
        String targetAttribute = "Survived";

        // Erstelle Printstream um trainierten Graphen in .dot File zu speichern
        PrintStream logStream;
        logStream = new PrintStream(new FileOutputStream("decision_tree.dot"));
        logStream.println("digraph G{");

        // Create new Classifier
        DecisionTreeClassifier dtc = new DecisionTreeClassifier(data.get("train"), attributes, logStream);

        // Discretize values
        dtc.registerDiscretization("Age", 8);
        dtc.registerDiscretization("Fare", 1);
        dtc.registerDiscretization("Parch", 6);

        // Trainiere den Baum
        dtc.trainDecisionTree(targetAttribute);

        // Ende der .dot File schreiben
        logStream.println("}");
        logStream.close();

        try {
            ArrayList<HashMap<String, String>> preds = dtc.predictDataset(data.get("test"), "Survived");
            CsvHelper.writeFile("./predictions.csv", preds);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
