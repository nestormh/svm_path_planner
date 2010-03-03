/**
 * 
 */
package sibtra.gps;

import java.util.Vector;

/**
 * Clase para almacenar los tramos generiados por {@link EditaFicherosRuta} y utilizados por
 * la gestión de flotas de Evelio.
 * 
 * @author alberto
 *
 */
public class Tramos {
	
	/** Representación de los datos de cada uno de los tramos */
	private class DatosTramo {
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

}
