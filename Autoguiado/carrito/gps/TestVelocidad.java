/*
 * TestVelocidad.java
 *
 * Created on 22 de mayo de 2007, 9:21
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package carrito.gps;

import carrito.server.serial.SerialConnection;
import java.io.*;

/**
 *
 * @author nestor
 */
public class TestVelocidad {
    long base = 0;
    
    SerialConnection sc = null;
    
    /** Creates a new instance of TestVelocidad */
    public TestVelocidad(String puerto) {
        sc = new SerialConnection(puerto);
    }
    
    public void test(int aceleracion, long segundos, long intervalo, String fichero) {
        System.out.println("Probando a " + aceleracion + " durante " + segundos + " segundos");
        try {
            PrintWriter pw = new PrintWriter(new File(fichero));
            base = System.currentTimeMillis();
            long tiempo = 0;
            while ((tiempo = System.currentTimeMillis() - base) < segundos) {
                sc.Avanza(aceleracion);
                pw.println(tiempo + "     " + sc.getVelocidad());
                System.out.println(tiempo + "     " + sc.getVelocidad());
                try {
                    Thread.sleep(intervalo);
                } catch (Exception ex) {}
            }
            pw.close();
        } catch(Exception e) {}              
    }
    
}
