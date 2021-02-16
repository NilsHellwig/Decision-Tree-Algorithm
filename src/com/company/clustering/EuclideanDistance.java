package com.company.clustering;

import java.util.Map;

public class EuclideanDistance {
    public double calculate(Map<String, Double> p1, Map<String, Double> p2) {
        double sum = 0;
        for (String key : p1.keySet()) {
            Double v1 = p1.get(key);
            Double v2 = p2.get(key);

            if (v1 != null && v2 != null) {
                sum += Math.pow(v1 - v2, 2);
            }
        }
        return Math.sqrt(sum);
    }
}
