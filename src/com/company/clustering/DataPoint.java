package com.company.clustering;

import java.util.Map;
import java.util.Objects;

public class DataPoint {
    private final String identifier;

    private final Map<String, Double> features;

    public DataPoint(Map<String, Double> features) {
        identifier = "";
        this.features = features;
    }

    public DataPoint(Map<String, Double> features, String identifier) {
        this.identifier = identifier;
        this.features = features;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        DataPoint dataPoint = (DataPoint) o;
        return identifier.equals(dataPoint.identifier) &&
                features.equals(dataPoint.features);
    }

    @Override
    public int hashCode() {
        return Objects.hash(identifier, features);
    }

    public String getIdentifier() {
        return identifier;
    }

    public Map<String, Double> getFeatures() {
        return features;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder().append(identifier).append("-> (");
        for(Map.Entry<String, Double> entry : features.entrySet()) {
            sb.append(entry.getValue()).append(", ");
        }
        sb.deleteCharAt(sb.toString().length()-1);
        sb.append(")");
        return sb.toString();
    }

    public static String toStringStatic(DataPoint d){
        return d.toString();
    }
}
