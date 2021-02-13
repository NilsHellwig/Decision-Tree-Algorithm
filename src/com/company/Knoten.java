package com.company;

public class Knoten {

    public String label;

    public Knoten(){
        System.out.println("Neuer Knoten");
        this.label = "";
    }

    public void setLabel(String label){
        this.label = label;
    }

    public String getLabel() {
        return this.label;
    }

}
