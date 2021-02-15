package com.company;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;

public class CsvHelper {
    public static ArrayList<HashMap<String,String>> readFile(String path){
        ArrayList<HashMap<String,String>> data = new ArrayList<HashMap<String,String>>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            int i = 0;
            String[] attributes = new String[0];
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


    public static void writeFile(String filePath, ArrayList<HashMap<String, String>> data) {
        writeFile(filePath, data, ';' ,"\n", 100);
    }

    public static void writeFile(String filePath, ArrayList<HashMap<String, String>> data, char delimiter) {
        writeFile(filePath, data, delimiter, "\n", 100);
    }

    public static void writeFile(String filePath, ArrayList<HashMap<String, String>> data, char delimiter, String newline) {
        writeFile(filePath, data, delimiter, newline, 100);
    }

    public static void writeFile(String filePath, ArrayList<HashMap<String, String>> data, char delimiter, String newline, int batchSize) {
        if (data.size() == 0) {
            return;
        }
        // write header
        StringBuilder sb = new StringBuilder();
        for (String s : data.get(0).keySet()) {
            sb.append(s).append(delimiter);
        }
        sb.append(newline);

        int count = 0;

        try (BufferedWriter br = new BufferedWriter(new FileWriter(filePath))) {
            for (HashMap<String, String> row : data) {
                count++;
                for (String s : row.keySet()) {
                    sb.append(row.get(s)).append(delimiter);
                }
                sb.append(newline);
                if (count % batchSize == 0) {
                    br.write(sb.toString());
                    sb = new StringBuilder();
                }
            }
            br.write(sb.toString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
