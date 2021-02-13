package com.company;

public class Knoten {

    public String label;
    public String attribute;
    public String value;

    public Knoten(){
        // System.out.println("Neuer Knoten");
        this.label = "";
        this.value = "";
        this.attribute = "";
    }

    public void setLabel(String label){
        this.label = label;
    }

    public String getLabel() {
        return this.label;
    }

    public void setAttribute(String attribute){
        this.attribute = attribute;
    }

    public String getAttribute() {
        return this.attribute;
    }

    public String getValue(){
        return this.value;
    }

    public void setValue(String value){
        this.value = value;
    }

}
