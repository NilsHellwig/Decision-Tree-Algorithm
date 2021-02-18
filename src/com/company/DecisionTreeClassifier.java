package com.company;

import com.company.clustering.Centroid;
import com.company.clustering.DataPoint;
import com.company.clustering.EuclideanDistance;
import com.company.clustering.KMeansClustering;

import java.io.PrintStream;
import java.util.*;

import static java.util.stream.Collectors.toSet;

public class DecisionTreeClassifier {

    private Knoten root;
    private ArrayList<HashMap<String, String>> dataset;
    private HashMap<String, ArrayList<String>> possibleValuesForAttributes;
    private HashMap<String, Integer> discretizations;
    private ArrayList<String> attributes;
    private HashMap<String, List<Centroid>> centroids;
    public PrintStream logStream;

    /**
     * Creates a decision tree classifier
     * @param trainingDataSetPath path to the training data set
     * @param targetAttribute name of the target attribute (column name)
     * @param attrs name of columns used for classification
     * @param logStream for writing lines in graph .dot file
     */
    public DecisionTreeClassifier(String trainingDataSetPath, String targetAttribute, ArrayList<String> attrs, PrintStream logStream) {
        dataset = CsvHelper.readFile(trainingDataSetPath);
        attributes = attrs;
        this.logStream = logStream;
        discretizations = new HashMap<String, Integer>();
        centroids = new HashMap<String, List<Centroid>>();
    }

    /**
     * Creates a decision tree classifier
     * @param trainingDataSetPath path to the training data set
     * @param targetAttribute name of the target attribute (column name)
     * @param attrs name of columns used for classification
     * @param logStream for writing lines in graph .dot file
     */
/*    public DecisionTreeClassifier(ArrayList<HashMap<String, String>> trainingData, String targetAttribute, ArrayList<String> attrs, PrintStream logStream) {
        dataset = trainingData;
        attributes = attrs;
        this.logStream = logStream;
        discretizations = new HashMap<String, Integer>();
        centroids = new HashMap<String, List<Centroid>>();
    }
*/
    /**
     * Predict the target attribute for an observation
     * @param obeservation Hash map containing the values of an observation
     * @return prediction result
     * @throws Exception if value that should be discretized is non numerical
     */
    public String predict(HashMap<String, String> obeservation) throws Exception {
        prepareObeservation(obeservation);
        return root.predict(obeservation);
    }

    /**
     * Prepares the observation by putting most common value in missing values and discretizing the specified numerical values
     * @param observation Hash map containing the values of an observation
     * @throws Exception  if value that should be discretized is non numerical
     */
    private void prepareObeservation(HashMap<String, String> observation) throws Exception {
        for(String columnName : discretizations.keySet()) {
            String value = observation.get(columnName);
            if("".equals(value) || value == null) {
                String mcv = mcv(columnName + "_____old", dataset, true);
                observation.put(columnName, mcv);
            }
            DataPoint d = EntryToDataPoint(observation, columnName, "newValue");
            List<Centroid> centroidList = centroids.get(columnName);
            //we dont know if the data has been seen before already, so instead we try to find the closest centroid to it
            //and use its nane as value
            Centroid c = KMeansClustering.findClosestCentroid(centroidList, d, new EuclideanDistance());
            replaceObservationValue(observation, columnName, c.getId());

        }

        for(String key : observation.keySet()){
            String value = observation.get(key);
            if(!discretizations.keySet().contains(value) && ("".equals(value) || value == null)){
                String mcv = mcv(key, dataset);
                observation.put(key, mcv);
            }
        }
    }

    /**
     * Actual Decision Tree training algorithm
     * @param targetAttribute specified target attribute
     */
    public void trainDecisionTree(String targetAttribute) {
        //apply discretization on the registered columns
        for(String key : discretizations.keySet()) {
            discretise(dataset, key, discretizations.get(key));
        }
        /* Diese Hashmap schaut anhand der Trainingsdaten, welche
         Werte ein Attribut überhaupt annehmen kann z.B. das Attribut Sex hat die Werte: male/female.
         Anhand der Trainingsdaten wird dies ermittelt. */
        getPossibleValuesForAttributes(dataset);

        // Prüfe, ob Laden der CSV-Datei funktioniert hat, durch ausgeben der ersten Zeile mit Daten
        System.out.println("<log> check dataset: " + dataset.get(0));

        // Erstelle den Startknoten
        root = lerne(dataset, targetAttribute, attributes);

    }

    private Knoten lerne(ArrayList<HashMap<String, String>> dataset, String targetAttribute, ArrayList<String> attributes) {
        // Schritt 2
        ArrayList<String> valuesOfTargetAttributeInDataset = new ArrayList<>();
        for (HashMap<String, String> datapoint : dataset) {
            valuesOfTargetAttributeInDataset.add(datapoint.get(targetAttribute));
        }
        HashSet<String> uniqueValuesFoundForTargetAttribute = new HashSet<String>();
        uniqueValuesFoundForTargetAttribute.addAll(valuesOfTargetAttributeInDataset);
        if (uniqueValuesFoundForTargetAttribute.size() == 1) {
            Knoten root = new Knoten();
            root.setLabel(uniqueValuesFoundForTargetAttribute.toArray()[0].toString());
            return root;
        }

        // Schritt 3
        if (attributes.size() == 0) {
            Knoten root = new Knoten();
            String label = mcv(targetAttribute, dataset);
            root.setLabel(label);
            return root;
        }

        // Schritt 4
        String bestAttribute = getAttributeWithHighestInformationGain(dataset, targetAttribute, attributes, possibleValuesForAttributes);

        // Schritt 5
        Knoten root = new Knoten();
        root.setAttribute(bestAttribute);

        // Schritt 6
        ArrayList<Knoten> children = new ArrayList<Knoten>();
        for (String possibleValue : possibleValuesForAttributes.get(bestAttribute)) {

            // Schritt 7
            Knoten rootChild;

            // Schritt 8
            ArrayList<HashMap<String, String>> datasetForChild = createSubSetOfData(dataset, bestAttribute, possibleValue);

            // Schritt 9
            if (datasetForChild.size() == 0) {
                rootChild = new Knoten();
                rootChild.setLabel(mcv(targetAttribute, dataset));
                rootChild.setValue(possibleValue);
                children.add(rootChild);
                // Schritt 10
            } else {
                ArrayList<String> attributesWithoutBestAttribute = new ArrayList<>();
                for (String att : attributes) {
                    if (att != bestAttribute) {
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
        for (Knoten child : children) {
            System.out.println("[ <Question: " + root.getAttribute() + "> <Value of Edge: " + root.getValue() + "> <Label: " + root.getLabel() + "> ] --> [ <Question: " + child.getAttribute() + "> < Value of Edge: " + child.getValue() + "> < Label: " + child.getLabel() + "> ]");
            if(child.getAttribute().equals("")){
                logStream.println("\""+root.getAttribute()+"\\n"+root.nodeId+"\" -> \"Prediction: "+child.getLabel()+"\\n"+child.nodeId+"\" [label=\""+child.getValue()+"\"];");
                logStream.println("\"Prediction: "+child.getLabel()+"\\n"+child.nodeId+"\" [shape=box, style=filled, color=red];");
                if(child.getLabel().equals("1")){
                    logStream.println("\"Prediction: "+child.getLabel()+"\\n"+child.nodeId+"\" [shape=box, style=filled, color=green];");
                }
            } else {
                logStream.println("\""+root.getAttribute()+"\\n"+root.nodeId+"\" -> \""+child.getAttribute()+"\\n"+child.nodeId+"\" [label=\""+child.getValue()+"\"];");
            }
        }

        return root;
    }

    private ArrayList<HashMap<String, String>> createSubSetOfData(ArrayList<HashMap<String, String>> dataset, String attributeToSplit, String valueOfAttributeToSplit) {
        ArrayList<HashMap<String, String>> filteredDataset = new ArrayList<HashMap<String, String>>();
        for (HashMap<String, String> datapoint : dataset) {
            if (datapoint.get(attributeToSplit).equals(valueOfAttributeToSplit)) {
                filteredDataset.add(datapoint);
            }
        }
        return filteredDataset;
    }

    private String mcv(String targetAttribute, ArrayList<HashMap<String, String>> dataset) {
        return mcv(targetAttribute, dataset, false);
    }

    /**
     * Get the most common value for an attribute
     * @param targetAttribute the specified attribute/column
     * @param dataset dataset to get the mcv from
     * @param ignoreEmptyOrNull false if empty/null values should be treated as actual value, true else
     * @return
     */
    private String mcv(String targetAttribute, ArrayList<HashMap<String,String>> dataset, boolean ignoreEmptyOrNull){
        HashMap<String,Integer> valueCounter = new HashMap<String,Integer>();
        for(HashMap<String,String> datapoint: dataset){

            String value = datapoint.get(targetAttribute);
            if(ignoreEmptyOrNull && (value == null || "".equals(value))) {
                continue;
            }

            if (valueCounter.containsKey(value)){
                valueCounter.put(value, valueCounter.get(value)+1);
            } else {
                valueCounter.put(value, 1);
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
            if (informationGains.get(attributeName) >= lowest_information_gain){
                lowest_information_gain = informationGains.get(attributeName);
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

    /**
     * predict based on a csv file
     * @param filePath path to the specified file
     * @return
     * @throws Exception if classifier is not trained
     */
    public ArrayList<HashMap<String, String>> predictCsv(String filePath) throws Exception {
        return predictCsv(filePath, null);
    }

    /**
     * predict and evaluate a csv file
     * @param filePath path to specified file
     * @param targetAttribute target attribute used for evaluation
     * @return
     * @throws Exception if classifier is not trained
     */
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
            try{
                String prediction = predict(observation);
                map.put("prediction", prediction);
                if(eval) {
                    if(map.get("prediction").equals(map.get(targetAttribute))) {
                        tp++;
                    }else{
                        fp++;
                    }
                }
            }catch(Exception e) {
                e.printStackTrace(System.err);
                System.out.println("Error during discretizing, skipping observation...");
            }
        }

        //calculate the precision
        if(eval) {
            double precision = tp/(tp+fp);
            System.out.println(new StringBuilder().append("Precision: ").append(precision).append(" / True Positive: ").append(tp).append(" / False Positive: ").append(fp));
        }

        return toPredict;

    }

    /**
     * Discretise a whole dataset by k-means-clustering
     * @param data dataset containing the values
     * @param columnName column to be discretized
     * @param clusterAmount k for k-means
     */
    private void discretise(ArrayList<HashMap<String, String>> data, String columnName, int clusterAmount) {
        if(data.size() == 0){
            return;
        }

        String mcv = mcv(columnName, data, true);
        ArrayList<DataPoint> dataPoints = new ArrayList<DataPoint>();

        //iterate over dataset
        for(int i = 0; i<data.size(); i++) {
            HashMap<String, String> row = data.get(i);
            if(row.get(columnName) == null || "".equals(row.get(columnName))){
                //put most common value if value is missing
                row.put(columnName, mcv);
            }
            String value = row.get(columnName);
            //if a non-int value appears, its non numerical
                if(!value.matches("(\\d+)|(\\d+.\\d+)")){
                System.out.println("Column contains non numerical data, returning...");
                System.out.println("Data:");
                System.out.println(value);
                return;
            }
            //turn the entry it into a datapoint
            dataPoints.add(EntryToDataPoint(row, columnName, (new StringBuilder("Node").append(i)).toString()));

        }
        //cluster the datapoints into k centroids
        Map<Centroid, List<DataPoint>> clusters = cluster(dataPoints, clusterAmount, 1000);
        //write the cluster names into the column used, write the column used into "[columnUser]_____old"
        reassignClusteredValues(data, clusters, columnName);
    }


    /**
     * clusters the data
     * @param data data to be clustered
     * @param k number of clusters
     * @param maxIterations number of iterations
     * @return map of centroids and their corresponding data points
     */
    private Map<Centroid, List<DataPoint>> cluster(List<DataPoint> data, int k, int maxIterations) {
        Map<Centroid, List<DataPoint>> clusters = KMeansClustering.fit(data, k, new EuclideanDistance(), maxIterations);
        return clusters;
    }

    /**
     * write the cluster names into the column used, write the column used into "[columnUsed]_____old"
     * @param data data to be adjusted
     * @param clusters map of clusters and their data points
     * @param columnName column to be replaced
     */
    private void reassignClusteredValues(ArrayList<HashMap<String, String>> data, Map<Centroid, List<DataPoint>> clusters, String columnName) {
        centroids.put(columnName, new ArrayList<Centroid>());
        clusters.forEach((key, value) -> {
            double clusterMinValue = Double.MAX_VALUE, clusterMaxValue = Double.MIN_VALUE;
            centroids.get(columnName).add(key);
            //find min/max values of entries in cluster first to correctly name the cluster later
            for(DataPoint d : value) {
                clusterMinValue = Math.min(clusterMinValue, d.getFeatures().get(columnName));
                clusterMaxValue = Math.max(clusterMaxValue, d.getFeatures().get(columnName));
           }
            String clusterName = new StringBuilder().append(columnName).append("[").append(clusterMinValue).append("-").append(clusterMaxValue).append("]").toString();
            key.setId(clusterName);

            for(DataPoint d : value) {
                clusterMinValue = Math.min(clusterMinValue, d.getFeatures().get(columnName));
                clusterMaxValue = Math.max(clusterMaxValue, d.getFeatures().get(columnName));
                int idx = Integer.parseInt(d.getIdentifier().replace("Node", ""));
                replaceObservationValue(data.get(idx), columnName, clusterName);
            }
        });
    }

    /**
     * Creates a one dimensional data point (1-d vector) for clustering
     * @param data observation
     * @param feature feature of the observation to be used
     * @param identifyingFeature a data point needs an identifying feature, so we can associate it back with the original entry
     * @return
     */
    private static DataPoint EntryToDataPoint(HashMap<String, String> data, String feature, String identifyingFeature) {
        Map<String, Double> coords = new HashMap<String, Double>();
        //this could be expanded so the data points contain more than one dimension, but thats not neccessary here
        double d = Double.parseDouble(data.get(feature));
        coords.put(feature, d);
        return new DataPoint(coords, identifyingFeature);
    }

    /**
     * replaces a value of an observation and backs it up into value_____old
     * @param observation observation containing values
     * @param columnName name of the value within the observation
     * @param newValue replacement of the value
     */
    private static void replaceObservationValue(HashMap<String, String> observation, String columnName, String newValue) {
        String valueOld = observation.get(columnName);
        observation.put(columnName + "_____old", valueOld);
        observation.put(columnName, newValue);
    }

    /**
     * registers a column of the dataset for discretization
     * @param column column name
     * @param k k for k-means clustering
     */
    public void registerDiscretization(String column, int k) {
        discretizations.put(column, k);
    }

    public static HashMap<String, ArrayList<HashMap<String, String>>> trainTestValidateSplit(ArrayList<HashMap<String, String>> dataset, double testSize, double validateSize) {
        HashMap<String, ArrayList<HashMap<String, String>>> splits = new HashMap<String, ArrayList<HashMap<String, String>>>();

        splits.put("test", new ArrayList<HashMap<String, String>>());
        splits.put("validate", new ArrayList<HashMap<String, String>>());

        int testAmount = (int) (testSize * dataset.size());
        int validateAmount = (int) (validateSize * dataset.size());

        Random r = new Random();
        for(int i = 0; i<testAmount; i++){
            int idx = r.nextInt(testAmount);
            splits.get("test").add(dataset.remove(idx));
        }

        for(int i = 0; i<validateAmount; i++){
            int idx = r.nextInt(validateAmount);
            splits.get("validate").add(dataset.remove(idx));
        }
        splits.put("train", dataset);

        System.out.println(new StringBuilder().append("Train: ").append(splits.get("train").size()).append(" Test:").append(splits.get("test").size()).append(" Validate:").append(splits.get("validate").size()).toString());

        return splits;
    }
}
