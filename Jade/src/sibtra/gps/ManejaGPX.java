package sibtra.gps;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import org.jdom.Document;
import org.jdom.Element;
import org.jdom.Namespace;
import org.jdom.output.Format;
import org.jdom.output.XMLOutputter;

/**
 * Clase para realizar la lectura y escritura de rutas en ficheros GPX
 * @author alberto
 *
 */
public class ManejaGPX {
	
	public static Namespace gpxNS=Namespace.getNamespace("http://www.topografix.com/GPX/1/1");
	public static Namespace xsiNS=Namespace.getNamespace("xsi","http://www.w3.org/2001/XMLSchema-instance");
	public static Namespace verNS=Namespace.getNamespace("ver","http://www.isaatc.ull.es/Verdino/1/0");


	/** 
	 * Salva la ruta pasada a el fichero indicado como GPX 
	 */
	public static void salvaAGPX(Ruta ra, File fich) {
		if(ra==null)
			throw new IllegalArgumentException("La ruta pasada no puede ser null");
		Document doc=new Document();
		Element root=new Element("gpx",ManejaGPX.gpxNS);
		doc.addContent(root);
		root.setAttribute("version", "1.1");
		root.setAttribute("creator", "ManejaGPX");
		root.setAttribute("schemaLocation","http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd",xsiNS);
		root.addNamespaceDeclaration(xsiNS);
		root.addNamespaceDeclaration(verNS);
		
		//Metadata
		Element mtdt=new Element("metadata",gpxNS);
		root.addContent(mtdt);
		mtdt.addContent(new Element("desc",gpxNS).setText("Volcado de Ruta"));
		
		//Punto del centro
		Element centro=puntoATrkpt(ra.getCentro());
		root.addContent(centro);
		centro.setName("wpt");
		centro.addContent(1,new Element("name",gpxNS).setText("Centro"));
		
		//Track
		Element trk=new Element("trk",gpxNS);
		root.addContent(trk);
		trk.addContent(new Element("name",gpxNS).setText("RutaUnica"));

		Element trkseg=new Element("trkseg",gpxNS);
		trk.addContent(trkseg);
		//pasamos a recorrer la ruta para añadir los puntos
		for(int i=0; i<ra.getNumPuntos(); i++) {
			GPSData pa=ra.getPunto(i);
			Element trkpt=puntoATrkpt(pa);
			trkseg.addContent(trkpt);
		}
		
		
		//Volcamos arbol DOM a fichero
		XMLOutputter oupt=new XMLOutputter(Format.getPrettyFormat());
		
		try {
			oupt.output(doc, new FileOutputStream(fich));
		} catch (Exception e) {
			System.err.println("No se pudo salvar documento:"+e.getMessage());
		}
	}
	
	private static Element puntoATrkpt(GPSData pa) {
		Element wpa=new Element("trkpt",gpxNS);
		wpa.setAttribute("lat", String.valueOf(pa.getLatitud()));
		wpa.setAttribute("lon", String.valueOf(pa.getLongitud()));
		wpa.addContent(new Element("ele",gpxNS).setText(String.valueOf(pa.getAltura())));
		wpa.addContent(new Element("sat",gpxNS).setText(String.valueOf(pa.getSatelites())));
		//Extensiones
		Element ext=new Element("extensions",gpxNS);
		wpa.addContent(ext);
		ext.addContent(new Element("nemea",verNS).setText(pa.getCadenaNMEA()));
		if(!Double.isNaN(pa.getVelocidad()))
			ext.addContent(new Element("vel",verNS).setText(String.valueOf(pa.getVelocidad())));
		if(pa.getAngulosIMU()!=null)
			ext.addContent(new Element("yaw",verNS).setText(String.valueOf(pa.getAngulosIMU().getYaw())));
		
		return wpa;
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String FicheroRuta="Rutas/Universidad/Par0525";
		Ruta rutaEspacial=null, rutaTemporal;
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(FicheroRuta));
			rutaEspacial=(Ruta)ois.readObject();
			rutaTemporal=(Ruta)ois.readObject();
			ois.close();
			salvaAGPX(rutaEspacial, new File("/tmp/prueba.gpx"));
			//salvaAGPX(rutaEspacial, new File("/dev/stdout"));
		} catch (IOException ioe) {
			System.err.println("Error al abrir el fichero " +FicheroRuta);
			System.err.println(ioe.getMessage());
		} catch (ClassNotFoundException cnfe) {
			System.err.println("Objeto leído inválido: " + cnfe.getMessage());            
		}


	}

}
