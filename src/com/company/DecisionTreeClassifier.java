package com.company;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;

public class DecisionTreeClassifier {

    private Knoten root;
    private ArrayList<HashMap<String,String>> dataset;
    private HashMap<String,ArrayList<String>> possibleValuesForAttributes;
    private ArrayList<String> attributes;

    public DecisionTreeClassifier(String trainingDataSetPath, String targetAttribute, ArrayList<String> attrs) {
        dataset = CsvHelper.readFile(trainingDataSetPath);
        attributes = attrs;
        // Trainiere den Baum
        trainDecisionTree(targetAttribute, dataset);
    }

    public String predict(HashMap<String, String> obeservation) {
        return root.predict(obeservation);
    }

    private void trainDecisionTree(String targetAttribute, ArrayList<HashMap<String,String>> dataset){
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
        // Schritt 2
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
                rootChild = new Knoten();
                rootChild.setLabel(mvc(targetAttribute, dataset));
                rootChild.setValue(possibleValue);
                children.add(rootChild);
            // Schritt 10
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
        double entropy_e = getEntropy(dataset, targetAttribute, possibleValuesForTargetAttribute);

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
            double informationGain = entropy_e-sum;
            informationGains.put(attribute, informationGain);
        }
        double lowest_information_gain = 0;
        for(String attributeName: informationGains.keySet()){
            if (informationGains.get(attributeName) > lowest_information_gain){
                lowest_information_gain = informationGains.get(attributeName);
                bestAttribute = attributeName;
            }
            /* Damit auch, wenn kein Attribut einen InformationGain > 0 hat die
               ganze Methode ein Attribut zurückgibt, wird auch der Fall == 0 geprüft */
            if (lowest_information_gain == 0){
                bestAttribute = attributeName;
            }
        }
        return bestAttribute;
    }

    private double getEntropy(ArrayList<HashMap<String,String>> dataset, String targetAttribute, ArrayList<String> possibleValuesForTargetAttribute){
        double entropy = 0.0;
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
            entropy+= partOfEntropy;
        }
        return entropy;
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

    public ArrayList<HashMap<String, String>> predictCsv(String filePath) throws Exception {
        return predictCsv(filePath, null);
    }

    public ArrayList<HashMap<String, String>> predictCsv(String filePath, String targetAttribute) throws Exception {
        if(root == null) {
            throw new Exception("Classifier is not trained yet");
        }
        boolean eval = targetAttribute != null;
        double tp = 0;
        double fp = 0;

        ArrayList<HashMap<String,String>> toPredict = CsvHelper.readFile(filePath);

        for(HashMap<String, String> map : toPredict) {
            HashMap<String, String> observation = new HashMap<String, String>();
            for(String s : attributes) {
                observation.put(s, map.get(s));
            }
            String prediction = root.predict(observation);
            map.put("prediction", prediction);

            if(eval) {
                if(map.get("prediction").equals(map.get(targetAttribute))) {
                    tp++;
                }else{
                    fp++;
                }
            }
        }

        if(eval) {
            double precision = tp/(tp+fp);
            System.out.println(new StringBuilder().append("Precision: ").append(precision).append(" / True Positive: ").append(tp).append(" / False Positive: ").append(fp));
        }

        return toPredict;

    }
}
