package com.company.clustering;

import java.util.Map;
import java.util.Objects;
import java.util.UUID;

public class Centroid {
    public static int clusterCount = 0;
    private final Map<String, Double> coordinates;
    private final String id;

    public Centroid(Map<String, Double> coordinates) {
        this.coordinates = coordinates;
        id = new StringBuilder().append("cluster").append(clusterCount++).toString();
    }

    public Map<String, Double> getCoordinates() {
        return coordinates;
    }

    public String getId(){
        return id;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Centroid centroid = (Centroid) o;
        return coordinates.equals(centroid.coordinates);
    }

    @Override
    public int hashCode() {
        return Objects.hash(coordinates);
    }
}
