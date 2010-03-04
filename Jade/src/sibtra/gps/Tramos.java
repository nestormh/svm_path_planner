/**
 * 
 */
package sibtra.gps;


import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Vector;

/**
 * Clase para almacenar los tramos generiados por {@link EditaFicherosRuta} y utilizados por
 * la gestión de flotas de Evelio.
 * 
 * @author alberto
 *
 */
public class Tramos implements Serializable {
	
	/**
	 * Número de serie. IMPORTANTE porque vamos a salvarlo en fichero directamente.
	 * Si cambiamos estructura del objeto tenemos que cambiar el número de serie y ver 
	 * como se cargan versiones anteriores.
	 * Para saber si es necesario cambiar el número ver 
	 *  http://java.sun.com/j2se/1.5.0/docs/guide/serialization/spec/version.html#9419
	 */
	private static final long serialVersionUID = 1L;

	/** Representación de los datos de cada uno de los tramos */
	private class DatosTramo implements Serializable {

		private static final long serialVersionUID = 1L;

		Ruta rt=null;
		String nombre=null;
		/** Deberá haber un elemento por cada una de las trayectorias en {@link EditaFicherosRuta#vecTramos}
		 * Si está en true indica que la trayectoria correspondiente es siguiente de esta
		 */
		Vector<Boolean> sig=new Vector<Boolean>();
		/** Deberá haber un elemento por cada una de las trayectorias en {@link EditaFicherosRuta#vecTramos}
		 * Si está en true indica que esta trayectoria tiene prioridad respeto a la correspondiente
		 */
		Vector<Boolean> prio=new Vector<Boolean>();
		/** Deberá haber un elemento por cada una de las trayectorias en {@link EditaFicherosRuta#vecTramos}
		 * Si está en true indica que esta trayectoria tiene prioridad en oposición respeto a la correspondiente
		 */
		Vector<Boolean> opo=new Vector<Boolean>();
		//el indice debe coincidir con el del PannelMuestraVariasTrayectorias
		//Resto de detalles lo sacamos del  PanelMuestraVariasTrayectorias
		DatosTramo(Ruta ruta, String nom) {
			rt=ruta;
			nombre=nom;
			//inicializamos vectores booleanos todos a false
			for(int i=0; i<vecTramos.size(); i++) {
				sig.add(false);
				prio.add(false);
				opo.add(false);
			}
		}
		
		public String toString() {
			String siguientes="Sig:";
			String prioridad="Prio:";
			String oposicion="Opo:";
			for(int i=0; i<sig.size(); i++) {
				if(sig.get(i)) siguientes+=" "+i;
				if(prio.get(i)) prioridad+=" "+i;
				if(opo.get(i)) oposicion+=" "+i;
			}
			return String.format("[%s| %s| %s| %s | %s ]"
					, nombre, rt.toString(), siguientes, prioridad, oposicion);
		}
	}

	private Vector<DatosTramo> vecTramos=new Vector<DatosTramo>();
	
	public int size() {
		return vecTramos.size();
	}
	
	/** @return la ruta de tramo i-ésimo */
	public Ruta getRuta(int i) {
		if(i<0 || i>=vecTramos.size())
			throw new IndexOutOfBoundsException("Numero de tramo ("+i+") incorrecto (sólo hay "+vecTramos.size()+")");
		return vecTramos.get(i).rt;
	}
	
	public String getNombre(int i) {
		if(i<0 || i>=vecTramos.size())
			throw new IndexOutOfBoundsException("Numero de tramo ("+i+") incorrecto (sólo hay "+vecTramos.size()+")");
		return vecTramos.get(i).nombre;		
	}
	
	public void setNombre(int i, String nombre) {
		if(i<0 || i>=vecTramos.size())
			throw new IndexOutOfBoundsException("Numero de tramo ("+i+") incorrecto (sólo hay "+vecTramos.size()+")");
		vecTramos.get(i).nombre=nombre;
	}

	public boolean isSiguiente(int ini, int sig) {
		if(ini<0 || ini>=vecTramos.size())
			throw new IndexOutOfBoundsException("Numero de tramo inicial ("+ini+") incorrecto (sólo hay "+vecTramos.size()+")");
		if(sig<0 || sig>=vecTramos.size())
			throw new IndexOutOfBoundsException("Numero de tramo siguiente ("+sig+") incorrecto (sólo hay "+vecTramos.size()+")");
		return vecTramos.get(ini).sig.get(sig);
	}

	public void setSiguiente(int ini, int sig, boolean valor) {
		if(ini<0 || ini>=vecTramos.size())
			throw new IndexOutOfBoundsException("Numero de tramo inicial ("+ini+") incorrecto (sólo hay "+vecTramos.size()+")");
		if(sig<0 || sig>=vecTramos.size())
			throw new IndexOutOfBoundsException("Numero de tramo siguiente ("+sig+") incorrecto (sólo hay "+vecTramos.size()+")");
		vecTramos.get(ini).sig.set(sig,valor);
	}

	public boolean isPrioritatio(int ini, int sig) {
		if(ini<0 || ini>=vecTramos.size())
			throw new IndexOutOfBoundsException("Numero de tramo inicial ("+ini+") incorrecto (sólo hay "+vecTramos.size()+")");
		if(sig<0 || sig>=vecTramos.size())
			throw new IndexOutOfBoundsException("Numero de tramo siguiente ("+sig+") incorrecto (sólo hay "+vecTramos.size()+")");
		return vecTramos.get(ini).prio.get(sig);
	}

	public void setPrioritario(int ini, int sig, boolean valor) {
		if(ini<0 || ini>=vecTramos.size())
			throw new IndexOutOfBoundsException("Numero de tramo inicial ("+ini+") incorrecto (sólo hay "+vecTramos.size()+")");
		if(sig<0 || sig>=vecTramos.size())
			throw new IndexOutOfBoundsException("Numero de tramo siguiente ("+sig+") incorrecto (sólo hay "+vecTramos.size()+")");
		vecTramos.get(ini).prio.set(sig,valor);
	}

	public boolean isPrioritarioOposicion(int ini, int sig) {
		if(ini<0 || ini>=vecTramos.size())
			throw new IndexOutOfBoundsException("Numero de tramo inicial ("+ini+") incorrecto (sólo hay "+vecTramos.size()+")");
		if(sig<0 || sig>=vecTramos.size())
			throw new IndexOutOfBoundsException("Numero de tramo siguiente ("+sig+") incorrecto (sólo hay "+vecTramos.size()+")");
		return vecTramos.get(ini).opo.get(sig);
	}

	public void setPrioritarioOposicion(int ini, int sig, boolean valor) {
		if(ini<0 || ini>=vecTramos.size())
			throw new IndexOutOfBoundsException("Numero de tramo inicial ("+ini+") incorrecto (sólo hay "+vecTramos.size()+")");
		if(sig<0 || sig>=vecTramos.size())
			throw new IndexOutOfBoundsException("Numero de tramo siguiente ("+sig+") incorrecto (sólo hay "+vecTramos.size()+")");
		vecTramos.get(ini).opo.set(sig,valor);
	}
	
	public void añadeTramo(Ruta ruta, String nombre) {
		if(ruta==null)
			throw new IllegalArgumentException("La ruta a añadir no pude ser null");
		if(vecTramos.size()>0) {
			//tenemos una primera ruta, usamos su centro
			ruta.actualizaSistemaLocal(vecTramos.get(0).rt);
			ruta.actualizaCoordenadasLocales();
		}
		vecTramos.add( new DatosTramo(ruta,nombre) );
		//añadimos un componenete al vector booleano de todos
		for(DatosTramo dra: vecTramos) {
			dra.sig.add(false);
			dra.prio.add(false);
			dra.opo.add(false);
		}
	}

	public void borraTramo(int i) {
		if(i<0 || i>=vecTramos.size())
			throw new IndexOutOfBoundsException("Numero de tramo ("+i+") incorrecto (sólo hay "+vecTramos.size()+")");
		vecTramos.remove(i);
		//borramos compoenente correspondiente en los vectores booleanos
		for(DatosTramo dra: vecTramos) {
			dra.sig.remove(i);
			dra.prio.remove(i);
			dra.opo.remove(i);
		}			
	}

	/** Hace que el tramo indicado pase a una posición anterior */
	public void subir(int i) {
		if(i<0 || i>=vecTramos.size())
			throw new IllegalArgumentException("Índice fuera de rango");
		if(i==0)
			return; //no hacemos nada
		DatosTramo dta=vecTramos.get(i);
		vecTramos.set(i,vecTramos.get(i-1));
		vecTramos.set(i-1, dta);
	}
	
	/** Hace que el tramo indicado pase a una posición posterior */
	public void bajar(int i) {
		if(i<0 || i>=vecTramos.size())
			throw new IllegalArgumentException("Índice fuera de rango");
		if(i==(vecTramos.size()-1))
			return; //no hacemos nada
		DatosTramo dta=vecTramos.get(i);
		vecTramos.set(i,vecTramos.get(i+1));
		vecTramos.set(i+1, dta);
	}
	
	
	public String toString() {
		String resultado="Tramos: "+vecTramos.size()+"\n";
		for(DatosTramo dta: vecTramos)
			resultado+=dta.toString()+"\n";
		return resultado;
	}
	
	/**
	 * @param file Fichero a cargar
	 * @return los tramos cargados. null si ha habido algún problema.
	 */
	public static Tramos cargaTramos(File file) {
		Tramos tramos=null;
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
			tramos=(Tramos)ois.readObject();
			ois.close();
		} catch (IOException ioe) {
			System.err.println("Error al abrir el fichero " + file.getName());
			System.err.println(ioe.getMessage());
			tramos=null;
		} catch (ClassNotFoundException cnfe) {
			System.err.println("Objeto leído inválido: " + cnfe.getMessage());
			tramos=null;
		}
//		System.out.println("Tramos cargados: "+tramos);
		return tramos;
	}
	
	/**
	 * Salva los tramos 
	 * en formato binario en el fichero indicado.
	 * @param tramos tramos a salvar
	 * @param fichero path del fichero
	 */
	public static boolean salvaTramos(Tramos tramos, String fichero) {
		try {
//			System.out.println("Salvando Tramos: "+tramos);
			File file = new File(fichero);
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file));
			oos.writeObject(tramos);
			oos.close();
			return true;
		} catch (IOException ioe) {
			System.err.println("Error al escribir en el fichero " + fichero);
			System.err.println(ioe.getMessage());
			return false;
		}
	}
	
}
