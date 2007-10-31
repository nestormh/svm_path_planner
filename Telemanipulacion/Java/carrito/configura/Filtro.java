/**
 * Paquete que contiene todas las clases descriptoras de las propiedades que
 * afectan a la aplicación
 */
package carrito.configura;

import java.io.*;

import javax.swing.filechooser.FileFilter;

/**
 * Clase que hace de filtro genérico para la selección de los ficheros con el
 * JFileChooser
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class Filtro extends FileFilter {
    /** Array que contiene todas las posibles extensiones */
    private String filtros[] = null;
    /** String que describe dichas extensiones */
    private String descripcion = null;

    /**
     * Constructor que inicializa las propiedades de la clase
     * @param filtros Array que contiene todas las posibles extensiones
     * @param descripcion String que describe dichas extensiones
     */
    public Filtro(String[] filtros, String descripcion) {
        this.filtros = filtros;
        this.descripcion = descripcion;
    }

    /**
     * Método que decide si un fichero pasa o no por el filtro, basándonos en la
     * extensión
     * @param file Fichero a comprobar
     * @return Devuelve true si es aceptado
     */
    public boolean accept(File file) {
        String filename = file.getName();
        for (int i = 0; i < filtros.length; i++) {
            if (filename.endsWith(filtros[i]))
                return true;
        }
        return false;
    }

    // Obtiene la descripción del filtro
    public String getDescription() {
        return descripcion;
    }
}
