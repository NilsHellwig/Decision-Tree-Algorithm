package com.company;
import java.io.*;
import java.util.*;


public class Main {

    public static void main(String[] args) {
        // Definiere Pfad der Traingsdaten
        String trainingDataSetPath = "src/dataset/train.csv";

        // Jede Hasmap repräsentiert eine Zeile im Datensatz für das Training
        ArrayList<HashMap<String,String>> dataset = getData(trainingDataSetPath);

        // Definiere, welche Attribute genutzt werden sollen, um das TargetAttribute vorauszusagen
        List<String> attributesList = Arrays.asList( "Pclass", "Sex", "SibSp","Age","Parch","Embarked");
        ArrayList<String> attributes = new ArrayList<>(attributesList);

        // Definiere, welches Attribut vorausgesagt werden soll
        String targetAttribute = "Survived";

        // "Trainiere" den Baum
        Knoten trainedTree = trainDecisionTree(trainingDataSetPath, targetAttribute, attributes, dataset);

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
        System.out.println(trainedTree.predict(example));

    }

    public static Knoten trainDecisionTree(String trainingDataSetPath, String targetAttribute, ArrayList<String> attributes, ArrayList<HashMap<String,String>> dataset){
        /* Diese Hashmap schaut anhand der Trainingsdaten, welche
         Werte ein Attribut überhaupt annehmen kann z.B. das Attribut Sex hat die Werte: male/female.
         Anhand der Trainingsdaten wird dies ermittelt. */
        HashMap<String,ArrayList<String>> possibleValuesForAttributes = getPossibleValuesForAttributes(dataset);

        // Prüfe, ob Laden der CSV-Datei funktioniert hat, durch ausgeben der ersten Zeile mit Daten
        System.out.println("<log> check dataset: "+dataset.get(0));

        // Erstelle den Startknoten
        Knoten root = lerne(dataset,targetAttribute,attributes, possibleValuesForAttributes);
        return root;
    }

    public static Knoten lerne(ArrayList<HashMap<String,String>> dataset, String targetAttribute, ArrayList<String> attributes, HashMap<String,ArrayList<String>> possibleValuesForAttributes){
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
                rootChild = lerne(datasetForChild, targetAttribute, attributesWithoutBestAttribute, possibleValuesForAttributes);
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

    public static ArrayList<HashMap<String,String>> createSubSetOfData(ArrayList<HashMap<String,String>> dataset, String attributeToSplit, String valueOfAttributeToSplit){
        ArrayList<HashMap<String,String>> filteredDataset = new ArrayList<HashMap<String,String>>();
        for (HashMap<String,String> datapoint: dataset){
            if (datapoint.get(attributeToSplit).equals(valueOfAttributeToSplit)){
                filteredDataset.add(datapoint);
            }
        }
        return filteredDataset;
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
        return Math.log(N) / Math.log(2.0);
    }

    public static HashMap<String,ArrayList<String>> getPossibleValuesForAttributes(ArrayList<HashMap<String,String>> dataset){
        HashMap<String,ArrayList<String>> possibleValuesForAttributes = new HashMap<String,ArrayList<String>>();
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
        return possibleValuesForAttributes;
    }

    public static ArrayList<HashMap<String,String>> getData(String path){
        ArrayList<HashMap<String,String>> data = new ArrayList<HashMap<String,String>>();
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
                    data.add(datapoint);
                }
                i++;
            }
        } catch (Exception e){
            System.out.println("loading dataset failed because of: "+e);
        }
        return data;
    }
}
