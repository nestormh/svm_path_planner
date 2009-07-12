/**
 * 
 */
package sibtra.ui.modulos;

import java.io.File;

import javax.swing.JFileChooser;
import javax.swing.JOptionPane;

import sibtra.gps.GPSData;
import sibtra.gps.Ruta;
import sibtra.gps.Trayectoria;
import sibtra.ui.VentanasMonitoriza;

/**
 * Moudulo para cargar ruta de fichero.
 * @author alberto
 *
 */
public class CargaRutaFichero implements SeleccionRuta {
	
	String NOMBRE="Carga Ruta";
	String DESCRIPCION="Carga ruta de un fichero de ruta";
	private JFileChooser jfc;
	private VentanasMonitoriza ventanaMonitoriza;
	private Ruta rutaEspacial;
	private double distMaxTr=0.1;

	/** No hace nada */
	public CargaRutaFichero() {
	}

	/** inicializa el módulo */
	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonitoriza) {
		ventanaMonitoriza=ventMonitoriza;
		jfc=new JFileChooser(new File("./Rutas"));
		return true;
	}

	/* (non-Javadoc)
	 * @see sibtra.ui.modulos.SeleccionRuta#getTrayectoria()
	 */
	public Trayectoria getTrayectoria() {
    	//necestamos leer archivo con la ruta
    	int devuelto = jfc.showOpenDialog(ventanaMonitoriza.ventanaPrincipal);
    	if (devuelto != JFileChooser.APPROVE_OPTION) {
    		//no se quiso seleccionar ruta
    		rutaEspacial=null;
    		return null;
    	}
    	ventanaMonitoriza.conexionGPS.loadRuta(jfc.getSelectedFile().getAbsolutePath());
    	if((rutaEspacial=ventanaMonitoriza.conexionGPS.getRutaEspacial())==null) {
    		JOptionPane.showMessageDialog(ventanaMonitoriza.ventanaPrincipal,
    				"No se cargó ruta adecuadamente de ese fichero",
    				"Error",
    				JOptionPane.ERROR_MESSAGE);
    		return null;
    	}

    	//tenemos ruta != null
        double desMag = rutaEspacial.getDesviacionM();
        System.out.println("Usando desviación magnética " + Math.toDegrees(desMag));
        ventanaMonitoriza.setDesviacionMagnetica(desMag);

        // MOstrar coodenadas del centro del sistema local
        GPSData centroToTr = rutaEspacial.getCentro();
        System.out.println("centro de la Ruta Espacial " + centroToTr);
        //Rellenamos la trayectoria con la nueva versión de toTr,que 
        //introduce puntos en la trayectoria de manera que la separación
        //entre dos puntos nunca sea mayor de la distMax
        Trayectoria Tr = new Trayectoria(rutaEspacial,distMaxTr);


        System.out.println("Longitud de la trayectoria=" + Tr.length());


		return Tr;
	}

	public String getDescripcion() {
		return DESCRIPCION;
	}

	public String getNombre() {
		return NOMBRE;
	}


	/** Perdemos puntero de {@link #jfc} :-( */
	public void terminar() {
		jfc=null;
	}

}
