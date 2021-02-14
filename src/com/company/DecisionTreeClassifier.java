package com.company;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;

public class DecisionTreeClassifier {

    private Knoten root;
    private ArrayList<HashMap<String,String>> dataset;
    private HashMap<String,ArrayList<String>> possibleValuesForAttributes;
    public DecisionTreeClassifier(String trainingDataSetPath, String targetAttribute, ArrayList<String> attributes) {

        getData(trainingDataSetPath);

        // "Trainiere" den Baum
        trainDecisionTree(trainingDataSetPath, targetAttribute, attributes, dataset);
    }

    public String predict(HashMap<String, String> obeservation) {
        return root.predict(obeservation);
    }

    private void trainDecisionTree(String trainingDataSetPath, String targetAttribute, ArrayList<String> attributes, ArrayList<HashMap<String,String>> dataset){
        /* Diese Hashmap schaut anhand der Trainingsdaten, welche
         Werte ein Attribut überhaupt annehmen kann z.B. das Attribut Sex hat die Werte: male/female.
         Anhand der Trainingsdaten wird dies ermittelt. */
        getPossibleValuesForAttributes(dataset);

        // Prüfe, ob Laden der CSV-Datei funktioniert hat, durch ausgeben der ersten Zeile mit Daten
        System.out.println("<log> check dataset: "+dataset.get(0));

        // Erstelle den Startknoten
        root = lerne(dataset,targetAttribute,attributes);

    }

    private Knoten lerne(ArrayList<HashMap<String,String>> dataset, String targetAttribute, ArrayList<String> attributes){
        // Schritt 1 und 2
        ArrayList<String> valuesOfTargetAttributeInDataset = new ArrayList<>();
        for(HashMap<String,String> datapoint: dataset){
            valuesOfTargetAttributeInDataset.add(datapoint.get(targetAttribute));
        }
        HashSet<String> uniqueValuesFoundForTargetAttribute = new HashSet<String>();
        uniqueValuesFoundForTargetAttribute.addAll(valuesOfTargetAttributeInDataset);
        if (uniqueValuesFoundForTargetAttribute.size() == 1){
            Knoten root = new Knoten();
            root.setLabel(uniqueValuesFoundForTargetAttribute.toArray()[0].toString());
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
        Knoten root = new Knoten();

        // Schritt 5
        root.setAttribute(bestAttribute);

        // Schritt 6
        ArrayList<Knoten> children = new ArrayList<Knoten>();
        for (String possibleValue: possibleValuesForAttributes.get(bestAttribute)){

            // Schritt 7
            Knoten rootChild;

            // Schritt 8
            ArrayList<HashMap<String,String>> datasetForChild = createSubSetOfData(dataset, bestAttribute, possibleValue);

            // Schritt 9
            if (datasetForChild.size() == 0){
                // Schritt 10 - ??
                rootChild = new Knoten();
                rootChild.setLabel(mvc(targetAttribute, dataset));
                rootChild.setValue(possibleValue);
                children.add(rootChild);
            } else {
                ArrayList<String> attributesWithoutBestAttribute = new ArrayList<>();
                for (String att: attributes){
                    if (att != bestAttribute){
                        attributesWithoutBestAttribute.add(att);
                    }
                }
                rootChild = lerne(datasetForChild, targetAttribute, attributesWithoutBestAttribute);
                rootChild.setValue(possibleValue);
                children.add(rootChild);
            }
        }

        // Füge die Children zu dem Parent hinzu
        root.setChildren(children);

        // Gebe neue Verbindungen aus
        for(Knoten child: children){
            System.out.println("[ <Question: "+root.getAttribute()+"> <Value of Edge: "+root.getValue()+"> <Label: "+root.getLabel()+"> ] --> [ <Question: "+child.getAttribute()+"> < Value of Edge: "+child.getValue()+"> < Label: "+child.getLabel()+"> ]");
        }

        return root;
    }

    private ArrayList<HashMap<String,String>> createSubSetOfData(ArrayList<HashMap<String,String>> dataset, String attributeToSplit, String valueOfAttributeToSplit){
        ArrayList<HashMap<String,String>> filteredDataset = new ArrayList<HashMap<String,String>>();
        for (HashMap<String,String> datapoint: dataset){
            if (datapoint.get(attributeToSplit).equals(valueOfAttributeToSplit)){
                filteredDataset.add(datapoint);
            }
        }
        return filteredDataset;
    }

    private String mvc(String targetAttribute, ArrayList<HashMap<String,String>> dataset){
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
            if (valueCounter.get(attributeKey) > mostFrequentValueCount){
                mostFrequentValueForAttribute = attributeKey;
                mostFrequentValueCount = valueCounter.get(attributeKey);
            }
        }
        return mostFrequentValueForAttribute;
    }

    private String getAttributeWithHighestInformationGain(ArrayList<HashMap<String,String>> dataset, String targetAttribute, ArrayList<String> attributes, HashMap<String,ArrayList<String>> possibleValuesForAttributes){
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
                double ev_e = (countOccuranceOfValueForAttribute / dataset.size())*getEntropy(filteredDataset,targetAttribute,possibleValuesForTargetAttribute);
                sum+=ev_e;
            }
            double informationGain = entropie_e-sum;
            informationGains.put(attribute, informationGain);
        }
        double lowest_Entropie = 0;
        for(String attributeName: informationGains.keySet()){
            if (informationGains.get(attributeName) > lowest_Entropie){
                lowest_Entropie = informationGains.get(attributeName);
                bestAttribute = attributeName;
            }
            if (lowest_Entropie == 0){
                bestAttribute = attributeName;
            }
        }
        return bestAttribute;
    }

    private double getEntropy(ArrayList<HashMap<String,String>> dataset, String targetAttribute, ArrayList<String> possibleValuesForTargetAttribute){
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

    private double log2(double N){
        return Math.log(N) / Math.log(2.0);
    }

    private void getPossibleValuesForAttributes(ArrayList<HashMap<String,String>> dataset){
        possibleValuesForAttributes = new HashMap<String,ArrayList<String>>();
        String[] attributes = dataset.get(0).keySet().toArray(new String[0]);
        for(int k  = 0; k < attributes.length; k++){
            ArrayList<String> possibleValuesForAttribute = new ArrayList<>();
            for (HashMap<String,String> data : dataset){
                if (data.get(attributes[k]).length() > 0){
                    possibleValuesForAttribute.add(data.get(attributes[k]));
                }
            }
            LinkedHashSet<String> hashSet = new LinkedHashSet<>(possibleValuesForAttribute);
            possibleValuesForAttribute = new ArrayList<>(hashSet);
            possibleValuesForAttribute.toArray(new String[0]);
            possibleValuesForAttributes.put(attributes[k],possibleValuesForAttribute);
        }
    }

    private void getData(String path){
        dataset = new ArrayList<HashMap<String,String>>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            int i = 0;
            String[] attributes = new String[13];
            while((line = br.readLine()) != null) {
                /* Quelle von diesem regulären Ausdruck: https://stackoverflow.com/questions/1757065/java-splitting-a-comma-separated-string-but-ignoring-commas-in-quotes
                 Dies wird geprüft, da es CSV-Dateien gibt, bei denen innerhalb von Anführungszeichen " " ein Komma (";") ist.
                 Dies kann zu Problemen führen, da ; der Delimiter ist. Für unseren gegebenen Datensatz ist dies aber keine Limitation */
                String[] tokens = line.split(";(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1);
                if(i==0){
                    attributes = tokens;
                } else {
                    HashMap<String, String> datapoint = new HashMap<String, String>();
                    for (int k = 0; k < attributes.length; k++) {
                        datapoint.put(attributes[k], tokens[k]);
                    }
                    dataset.add(datapoint);
                }
                i++;
            }
        } catch (Exception e){
            System.out.println("loading dataset failed because of: "+e);
        }
    }
}
