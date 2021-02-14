package com.company;
import java.io.*;
import java.util.*;


public class Main {

    public static void main(String[] args) {
        // Definiere Pfad der Traingsdaten
        String trainingDataSetPath = "src/dataset/train.csv";

        // Definiere, welche Attribute genutzt werden sollen, um das TargetAttribute vorauszusagen
        List<String> attributesList = Arrays.asList( "Pclass", "Sex", "SibSp","Age","Parch","Embarked");
        ArrayList<String> attributes = new ArrayList<>(attributesList);

        // Definiere, welches Attribut vorausgesagt werden soll
        String targetAttribute = "Survived";

        DecisionTreeClassifier dtc = new DecisionTreeClassifier(trainingDataSetPath, targetAttribute, attributes);


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


    }

}
