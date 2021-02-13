package com.company;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.nio.file.Path;

public class Main {

    public static void main(String[] args) {
        // Definiere Pfade der Datensätze
        String training_path = "src/dataset/functionality test/train.csv";
        String test_path = "src/dataset/test.csv";

        /* Erstelle ArrayList mit Beispieldatenpunkten, die als Hashmap repräsentiert
         werden. */
        ArrayList<HashMap<String,String>> dataset = getData(training_path);

        /* Diese Hashmap schaut anhand der Trainingsdaten, welche
         Werte ein Attribut überhaupt annehmen kann z.B. das Attribut Sex hat die Werte: male/female.
         Anhand der Trainingsdaten wird dies ermittelt. */
        HashMap<String,ArrayList<String>> possibleValuesForAttributes = getPossibleValuesForAttributes(dataset);

        // Definiere unsere Attribute zum Vorhersagen des targetAttribute
        ArrayList<String> attributes = new ArrayList<String>();
        attributes.add("outlook");
        attributes.add("humidity");
        attributes.add("wind");

        // Attribut, dessen Wert vorausgesagt werden soll
        String targetAttribute = "tennis";

        // Prüfe, ob Laden der CSV-Datei funktioniert hat.
        System.out.println("<log> check dataset: "+dataset);

        // Erstelle Wurzelknoten
        Knoten root = lerne(dataset,targetAttribute,attributes, possibleValuesForAttributes);
    }

    public static Knoten lerne(ArrayList<HashMap<String,String>> dataset, String targetAttribute, ArrayList<String> attributes, HashMap<String,ArrayList<String>> possibleValuesForAttributes){
        // Schritt 1/2
        ArrayList<String> valuesOfTargetAttributeInDataset = new ArrayList<>();
        for(HashMap<String,String> datapoint: dataset){
            valuesOfTargetAttributeInDataset.add(datapoint.get(targetAttribute));
        }
        HashSet<String> uniqueValuesFoundForTargetAttribute = new HashSet<String>();
        uniqueValuesFoundForTargetAttribute.addAll(valuesOfTargetAttributeInDataset);
        if (uniqueValuesFoundForTargetAttribute.size() == 1){
            Knoten root = new Knoten();
            root.setLabel(uniqueValuesFoundForTargetAttribute.toArray()[0].toString());
            System.out.println(root.getLabel());
            return root;
        }

        // Schritt 3
        if (attributes.size() == 0){
            Knoten root = new Knoten();
            String label = mvc(targetAttribute, dataset);
            root.setLabel(label);
            return root;
        }

        // Schritt 4
        String bestAttribute = getAttributeWithHighestInformationGain(dataset, targetAttribute, attributes, possibleValuesForAttributes);
        System.out.println(bestAttribute);



        return new Knoten();
    }

    public static String mvc(String targetAttribute, ArrayList<HashMap<String,String>> dataset){
        HashMap<String,Integer> valueCounter = new HashMap<String,Integer>();
        for(HashMap<String,String> datapoint: dataset){
            if (valueCounter.containsKey(datapoint.get(targetAttribute))){
                valueCounter.put(datapoint.get(targetAttribute), valueCounter.get(datapoint.get(targetAttribute))+1);
            } else {
                valueCounter.put(datapoint.get(targetAttribute),1);
            }
        }
        String mostFrequentValueForAttribute = "";
        int mostFrequentValueCount = 0;
        for (String attributeKey: valueCounter.keySet()){
            System.out.println(attributeKey+" "+valueCounter.get(attributeKey));
            if (valueCounter.get(attributeKey) > mostFrequentValueCount){
                mostFrequentValueForAttribute = attributeKey;
                mostFrequentValueCount = valueCounter.get(attributeKey);
            }
        }
        return mostFrequentValueForAttribute;
    }

    public static String getAttributeWithHighestInformationGain(ArrayList<HashMap<String,String>> dataset, String targetAttribute, ArrayList<String> attributes, HashMap<String,ArrayList<String>> possibleValuesForAttributes){
        String bestAttribute = "";
        HashMap<String,Double> informationGains = new HashMap<String,Double>();
        ArrayList<String> possibleValuesForTargetAttribute = possibleValuesForAttributes.get(targetAttribute);

        // Zunächst wird die Entropie(E) berechnet
        double entropie_e = getEntropy(dataset, targetAttribute, possibleValuesForTargetAttribute);

        // für jedes Attribute wird nun der Information Gain berechnet
        for(String attribute : attributes){
                // Dann wird der zweite Teil der Formel berechnet (Summe über Werte des Attributs)
                ArrayList<String> possibleValuesForAttribute = possibleValuesForAttributes.get(attribute);
                double sum = 0.0;
                for (String value: possibleValuesForAttribute){
                    double countOccuranceOfValueForAttribute = 0.0;
                    ArrayList<HashMap<String,String>> filteredDataset = new ArrayList<HashMap<String,String>>();
                    for (HashMap<String,String> datapoint: dataset){
                        if (datapoint.get(attribute).equals(value)){
                            countOccuranceOfValueForAttribute += 1;
                            filteredDataset.add(datapoint);
                        }
                    }
                    //System.out.println(attribut+"------");
                    //System.out.println(getEntropy(filteredDataset,attribut,moeglicheWerteFürZielLabel));
                    //System.out.println("targetAtt "+attribute+" "+value+" "+getEntropy(filteredDataset,targetAttribute,possibleValuesForTargetAttribute));
                    double ev_e = (countOccuranceOfValueForAttribute / dataset.size())*getEntropy(filteredDataset,targetAttribute,possibleValuesForTargetAttribute);
                    sum+=ev_e;
                }
                //System.out.println(attribute);
                //System.out.println(entropie_e+" "+sum);
                double informationGain = entropie_e-sum;
                informationGains.put(attribute, informationGain);
        }
        double lowest_Entropie = 0;
        for(String attributeName: informationGains.keySet()){
            System.out.println("Gain for "+attributeName+" "+informationGains.get(attributeName));
            if (informationGains.get(attributeName) > lowest_Entropie){
                lowest_Entropie = informationGains.get(attributeName);
                bestAttribute = attributeName;
            }
        }
        System.out.println("<log> best Attribute is "+bestAttribute);
        return bestAttribute;
    }

    public static double getEntropy(ArrayList<HashMap<String,String>> dataset, String targetAttribute, ArrayList<String> possibleValuesForTargetAttribute){
        double entropie = 0.0;
        for (String attributeValue : possibleValuesForTargetAttribute){
            double countOccuranceOfValueForZielLabel = 0.0;
            for (HashMap<String,String> datapoint: dataset){
                if (datapoint.get(targetAttribute).equals(attributeValue)){
                    countOccuranceOfValueForZielLabel += 1;
                }
            }
            double partOfEntropy;
            if (countOccuranceOfValueForZielLabel == 0.0){
                partOfEntropy = 0;
            } else {
                partOfEntropy = -(countOccuranceOfValueForZielLabel/dataset.size())*log2(countOccuranceOfValueForZielLabel/dataset.size());
            }
            entropie+= partOfEntropy;
        }
        return entropie;
    }

    public static double log2(double N){
        double result = Math.log(N) / Math.log(2.0);
        return result;
    }

    public static HashMap<String,ArrayList<String>> getPossibleValuesForAttributes(ArrayList<HashMap<String,String>> dataset){
        HashMap<String,ArrayList<String>> possibleValuesForAttributes = new HashMap<String,ArrayList<String>>();
        String[] attributes = dataset.get(0).keySet().toArray(new String[0]);
        for(int k  = 0; k < attributes.length; k++){
            ArrayList<String> possibleValuesForAttribute = new ArrayList<>();
            for (HashMap<String,String> data : dataset){
                /* regex would be possible too: data.get(attributes[k]).matches("[a-zA-Z0-9]+") &&*/
                if (data.get(attributes[k]).length() > 0){
                    possibleValuesForAttribute.add(data.get(attributes[k]));
                }
            }
            LinkedHashSet<String> hashSet = new LinkedHashSet<>(possibleValuesForAttribute);
            possibleValuesForAttribute = new ArrayList<>(hashSet);
            possibleValuesForAttribute.toArray(new String[0]);
            possibleValuesForAttributes.put(attributes[k],possibleValuesForAttribute);
        }
        return possibleValuesForAttributes;
    }

    public static ArrayList<HashMap<String,String>> getData(String path){
        ArrayList<HashMap<String,String>> data = new ArrayList<HashMap<String,String>>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            int i = 0;
            String[] attributes = new String[13];
            while((line = br.readLine()) != null) {
                // Source for regexp: https://stackoverflow.com/questions/1757065/java-splitting-a-comma-separated-string-but-ignoring-commas-in-quotes
                String[] tokens = line.split(";(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1);
                if(i==0){
                    attributes = tokens;
                } else {
                    HashMap<String, String> datapoint = new HashMap<String, String>();
                    for (int k = 0; k < attributes.length; k++) {
                        datapoint.put(attributes[k], tokens[k]);
                    }
                    data.add(datapoint);
                }
                i++;
            }
        } catch (Exception e){
            System.out.println(e);
        }
        return data;
    }
}
