package com.company;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.UUID;

public class Knoten {

    public String label;
    public String attribute;
    public String value;
    public ArrayList<Knoten> children;
    public String nodeId;

    public Knoten(){
        this.label = "";
        this.value = "";
        this.attribute = "";

        /* Um bei Graphviz mehrere Knoten, die das selbe Attribut besitzen
           auseinanderhalten zu können, wird für jeden Knoten zusätzlich noch eine
           eindeutige Id (nodeId) (8-stellig) erstellt bei der Initialisierung*/
        createNodeId();
    }

    public void createNodeId(){
        this.nodeId  = UUID.randomUUID().toString().substring(0, 8);
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

    public void setChildren(ArrayList<Knoten> children){
        this.children = children;
    }

    public String predict(HashMap<String, String> example){
        if (!label.equals("")){
            return label;
        }
        for (String attributeKey: example.keySet()){
            if (attribute.equals(attributeKey)){
                for (Knoten child: children){
                    if (child.value.equals(example.get(attributeKey))){
                        if (!child.getLabel().equals("")){
                            System.out.println(getAttribute()+" --"+child.getValue()+"--> <"+child.getLabel()+">");
                        } else {
                            System.out.println(getAttribute() + " ---"+child.getValue()+"--> " + child.getAttribute());
                        }
                        return child.predict(example);
                    }
                }
            }
        }
        return "no label found error";
    }

}
