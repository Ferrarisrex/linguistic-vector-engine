package com.ferra;

import org.graalvm.polyglot.*;
import java.io.*;

public class BinToTxtConverter {
    public static void main(String[] args) {
        String binFile = "GoogleNews-vectors-negative300.bin";
        String txtFile = "GoogleNews-vectors-negative300.txt";

        try (Context context = Context.newBuilder("python")
                                      .allowAllAccess(true)
                                      .build()) {

            // Python code as a string
            String pyCode = ""
                + "from gensim.models import KeyedVectors\n"
                + "print('Loading Word2Vec binary...')\n"
                + "model = KeyedVectors.load_word2vec_format('" + binFile + "', binary=True)\n"
                + "print('Saving as text file...')\n"
                + "model.wv.save_word2vec_format('" + txtFile + "', binary=False)\n"
                + "print('Done!')\n";

            // Run Python code in-process
            context.eval("python", pyCode);

        } catch (PolyglotException e) {
            System.err.println("Polyglot error: " + e.getMessage());
        }
    }
}
