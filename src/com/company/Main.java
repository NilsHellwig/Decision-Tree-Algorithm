package com.company;
import java.io.*;
import java.util.*;
import java.io.PrintStream;
import java.io.FileOutputStream;

public class Main {

    public static void main(String[] args) throws FileNotFoundException {
        // Definiere Pfad der Traingsdaten
        String trainingDataSetPath = "src/dataset/train.csv";

        // Definiere, welche Attribute genutzt werden sollen, um das TargetAttribute vorauszusagen
        List<String> attributesList = Arrays.asList( "Pclass", "Sex", "SibSp","Age","Parch","Embarked");
        ArrayList<String> attributes = new ArrayList<>(attributesList);

        // Definiere, welches Attribut vorausgesagt werden soll
        String targetAttribute = "Survived";
        // Test PrintStream
        PrintStream logStream;

        logStream = new PrintStream(new FileOutputStream("decision_tree.dot"));
        logStream.println("digraph G{");
        DecisionTreeClassifier dtc = new DecisionTreeClassifier(trainingDataSetPath, targetAttribute, attributes, logStream);
        logStream.println("}");
        logStream.close();
        dtc.registerDiscretization("Age", 8);

        // Trainiere den Baum
        dtc.trainDecisionTree(targetAttribute);

        /*
        // Teste den Baum anhand eines Beispiels
        HashMap<String, String> example = new HashMap<String, String>()
        {{
            put("Pclass", "3");
            put("Sex", "male");
            put("SibSp", "1");
            put("Age", "22");
            put("Parch", "0");
            put("Embarked", "S");
        }};

        System.out.println(dtc.predict(example));
         */

        try {
            ArrayList<HashMap<String, String>> preds = dtc.predictCsv(trainingDataSetPath, "Survived");
            CsvHelper.writeFile("./predictions.csv", preds);
        }catch(Exception e){
            e.printStackTrace();
        }
    }

}
