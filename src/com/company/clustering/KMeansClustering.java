package com.company.clustering;

import javax.xml.crypto.Data;
import java.util.*;

import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toSet;

//example for k-means-clustering taken from here https://www.baeldung.com/java-k-means-clustering-algorithm

public class KMeansClustering {

    private static final Random random = new Random();

    public static Map<Centroid, List<DataPoint>> fit(List<DataPoint> dataPoints,
                                                  int k,
                                                  EuclideanDistance distance,
                                                  int maxIterations) {

        List<Centroid> centroids = randomCentroids(dataPoints, k);
        Map<Centroid, List<DataPoint>> clusters = new HashMap<>();
        Map<Centroid, List<DataPoint>> lastState = new HashMap<>();

        // iterate for a pre-defined number of times
        for (int i = 0; i < maxIterations; i++) {
            boolean isLastIteration = i == maxIterations - 1;

            // in each iteration we should find the nearest centroid for each record
            for (DataPoint record : dataPoints) {
                Centroid centroid = nearestCentroid(record, centroids, distance);
                assignToCluster(clusters, record, centroid);
            }

            // if the assignments do not change, then the algorithm terminates
            boolean shouldTerminate = isLastIteration || clusters.equals(lastState);
            lastState = clusters;
            if (shouldTerminate) {
                break;
            }

            // at the end of each iteration we should relocate the centroids
            centroids = relocateCentroids(clusters);
            clusters = new HashMap<>();
        }

        return lastState;
    }

    private static List<Centroid> randomCentroids(List<DataPoint> data, int k) {
        List<Centroid> centroids = new ArrayList<>();
        Map<String, Double> maxs = new HashMap<>();
        Map<String, Double> mins = new HashMap<>();

        for (DataPoint record : data) {
            record.getFeatures().forEach((key, value) -> {
                // compares the value with the current max and choose the bigger value between them
                maxs.compute(key, (k1, max) -> max == null || value > max ? value : max);

                // compare the value with the current min and choose the smaller value between them
                mins.compute(key, (k1, min) -> min == null || value < min ? value : min);
            });
        }

        Set<String> attributes = data.stream()
                .flatMap(e -> e.getFeatures().keySet().stream())
                .collect(toSet());
        for (int i = 0; i < k; i++) {
            Map<String, Double> coordinates = new HashMap<>();
            for (String attribute : attributes) {
                double max = maxs.get(attribute);
                double min = mins.get(attribute);
                coordinates.put(attribute, random.nextDouble() * (max - min) + min);
            }

            centroids.add(new Centroid(coordinates));
        }

        return centroids;
    }

    private static Centroid nearestCentroid(DataPoint dataPoint, List<Centroid> centroids, EuclideanDistance distance) {
        double minimumDistance = Double.MAX_VALUE;
        Centroid nearest = null;

        for (Centroid centroid : centroids) {
            double currentDistance = distance.calculate(dataPoint.getFeatures(), centroid.getCoordinates());

            if (currentDistance < minimumDistance) {
                minimumDistance = currentDistance;
                nearest = centroid;
            }
        }

        return nearest;
    }

    private static void assignToCluster(Map<Centroid, List<DataPoint>> clusters,
                                        DataPoint dataPoint,
                                        Centroid centroid) {
        clusters.compute(centroid, (key, list) -> {
            if (list == null) {
                list = new ArrayList<>();
            }

            list.add(dataPoint);
            return list;
        });
    }

    private static Centroid average(Centroid centroid, List<DataPoint> dataPoints) {
        if (dataPoints == null || dataPoints.isEmpty()) {
            return centroid;
        }

        Map<String, Double> average = centroid.getCoordinates();
        dataPoints.stream().flatMap(e -> e.getFeatures().keySet().stream())
                .forEach(k -> average.put(k, 0.0));

        for (DataPoint dataPoint : dataPoints) {
            dataPoint.getFeatures().forEach(
                    (k, v) -> average.compute(k, (k1, currentValue) -> v + currentValue)
            );
        }

        average.forEach((k, v) -> average.put(k, v / dataPoints.size()));

        return new Centroid(average);
    }

    private static List<Centroid> relocateCentroids(Map<Centroid, List<DataPoint>> clusters) {
        return clusters.entrySet().stream().map(e -> average(e.getKey(), e.getValue())).collect(toList());
    }

}
