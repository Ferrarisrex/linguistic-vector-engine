package com.ferra;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class App {
    private static final String EMBEDDING = "dolma_300_2024_1.2M.100_combined.txt";
    private static final int VECTOR_SIZE = 300; // must match your file dimension
    private static final Map<String, double[]> embeddings = new HashMap<>();

    public static void main(String[] args) throws Exception {
        System.out.println("Loading embeddings...");
        loadEmbeddings(EMBEDDING);

        System.out.println("Loaded " + embeddings.size() + " words.");
        System.out.println("LVE - Linguistic Vector Engine");
        System.out.println("Type expressions like: king - man + woman");
        System.out.println("Type 'exit' to quit.");

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("> ");
            String line = scanner.nextLine().trim();
            if (line.equalsIgnoreCase("exit")) break;
            if (line.isEmpty()) continue;

            try {
                double[] result = computeExpression(line);
                // build exclusion set of input tokens (lowercased)
                Set<String> exclude = Arrays.stream(line.split("\\s+"))
                        .map(t -> t.replaceAll("[^A-Za-z0-9_\\-']", "").toLowerCase())
                        .filter(s -> !s.isEmpty())
                        .collect(Collectors.toSet());

                List<String> nearest = findNearest(result, 5, exclude);
                System.out.println("Nearest words: " + nearest);
            } catch (RuntimeException e) {
                System.out.println("Error: " + e.getMessage());
            }
        }
    }

    private static void loadEmbeddings(String filePath) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            long count = 0;
            while ((line = br.readLine()) != null) {
                String[] parts = line.trim().split("\\s+");
                if (parts.length < VECTOR_SIZE + 1) continue;
                String word = parts[0].toLowerCase();
                double[] vec = new double[VECTOR_SIZE];
                for (int i = 0; i < VECTOR_SIZE; i++) {
                    vec[i] = Double.parseDouble(parts[i + 1]);
                }
                // L2-normalize immediately (so cosine -> dot)
                double norm = 0;
                for (double v : vec) norm += v * v;
                norm = Math.sqrt(norm);
                if (norm > 0) {
                    for (int i = 0; i < VECTOR_SIZE; i++) vec[i] /= norm;
                }
                embeddings.put(word, vec);
                count++;
            }
            System.out.println("Read " + count + " lines from " + filePath);
        }
    }

    // compute expression like "king - man + woman"
    private static double[] computeExpression(String expr) {
        String[] tokens = expr.split("\\s+");
        double[] result = new double[VECTOR_SIZE]; // start at zero
        String op = "+";
        boolean any = false;

        for (String raw : tokens) {
            if (raw.equals("+") || raw.equals("-")) {
                op = raw;
                continue;
            }
            String token = raw.replaceAll("[^A-Za-z0-9_\\-']", "").toLowerCase();
            if (token.isEmpty()) continue;
            double[] vec = embeddings.get(token);
            if (vec == null) throw new RuntimeException("Word not found: " + token);
            any = true;
            int sign = op.equals("+") ? 1 : -1;
            for (int i = 0; i < VECTOR_SIZE; i++) {
                result[i] += sign * vec[i];
            }
        }
        if (!any) throw new RuntimeException("No valid words in expression.");

        // normalize the result (so cosine = dot product)
        double norm = 0;
        for (double v : result) norm += v * v;
        norm = Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < VECTOR_SIZE; i++) result[i] /= norm;
        }
        return result;
    }

    // find nearest (embeddings are normalized) -- exclude input words
    private static List<String> findNearest(double[] target, int topN, Set<String> exclude) {
        PriorityQueue<Map.Entry<String, Double>> pq = new PriorityQueue<>(
                Comparator.comparingDouble(Map.Entry::getValue)
        );

        for (Map.Entry<String, double[]> entry : embeddings.entrySet()) {
            String word = entry.getKey();
            if (exclude != null && exclude.contains(word)) continue;
            double sim = dot(target, entry.getValue()); // dot == cosine (both normalized)
            pq.offer(new AbstractMap.SimpleEntry<>(word, sim));
            if (pq.size() > topN) pq.poll();
        }

        List<Map.Entry<String, Double>> list = new ArrayList<>();
        while (!pq.isEmpty()) list.add(pq.poll());
        Collections.reverse(list); // highest first
        return list.stream().map(Map.Entry::getKey).collect(Collectors.toList());
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }
}
