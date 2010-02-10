/**
 * 
 */
package sibtra.util;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;

/**
 * Clase para todas aquellas etiquetas cuyo valor se actualiza invocando 
 * un método sin argumentos del objeto pasado
 * @author alberto
 *
 */
@SuppressWarnings("serial")
public class LabelDatoFormato extends LabelDato {

	/** metodo que invocar */
	Method 	metodo;
	/** el método es estatico de la clase */
	boolean metodoEstatico;
	
	/** formato a aplicar */
	String 	formato;
	/** clase del objeto sobre el que invocar el método */
	Class	clase;
	
	protected LabelDatoFormato copiaResumen=null;

	
	/**
	 * Crea etiqueta con texto inicial
	 * @param clase clase cuyo método vamos a invocar
	 * @param nomMetodo nombre del metodo a invocar en cada actulización
	 * @param formato formato de {@link String.format} que se utilizara en cada actulización
	 * @param textoInicial
	 */
	public LabelDatoFormato(Class clase, String nomMetodo, String formato, String textoInicial) {
		this(clase,nomMetodo,formato);
		setText(textoInicial);
	}
	
	/**
	 * Crea etiqueta con texto inicial aplicandoles el formato a valor inicial por defecto (0)
	 * @param clase clase cuyo método vamos a invocar
	 * @param nomMetodo nombre del metodo a invocar en cada actulización
	 * @param formato formato de {@link String.format} que se utilizara en cada actulización
	 */
	public LabelDatoFormato(Class clase, String nomMetodo, String formato) {
		super("######");
		this.formato=formato;
		this.clase=clase;
		if(clase==null)
			throw new IllegalArgumentException("La clase pasada ha de ser != de null");
		try {
			this.metodo=clase.getMethod(nomMetodo,(Class[])null);
			metodoEstatico=Modifier.isStatic(metodo.getModifiers());
		} catch (SecurityException e) {
			throw new IllegalArgumentException("El metodo "+nomMetodo
					+ " no se puede invocar por razones de seguridad");
		} catch (NoSuchMethodException e) {
			throw new IllegalArgumentException("El metodo "+nomMetodo
					+ " no existe sin argumentos en la calse "
					+ clase.getCanonicalName());			
		}
		//Vamos a ver de que tipo es el valor retornado
		try {
			Class tipoRetorno=metodo.getReturnType();
			if(tipoRetorno.equals(Double.TYPE)){
				setText(String.format(formato, (double)0.0));
			} else if(tipoRetorno.equals(Integer.TYPE)) {
				setText(String.format(formato, (int)0));				
			} else if(tipoRetorno.equals(Long.TYPE)) {
				setText(String.format(formato, (long)0));				
			}
		} catch (Exception e) {
			//dejamos el texto por defecto
		}
	}
	
	public LabelDatoFormato(LabelDatoFormato actual) {
		super("######");
		clase=actual.clase;
		formato=actual.formato;
		metodo=actual.metodo;
	}

	/**
	 * Actuliza la etiqueta invocando al método correspondiente del objeto pasado.
	 * Se comprueba que le objetosea de la clase correcta
	 * @param objeto sobre el que invocar el metodo
	 * @param hayCambio si hay nuevo dato
	 */
	public void Actualiza(Object objeto,boolean hayCambio) {
		setEnabled(hayCambio);
		if(hayCambio) {
			try {
				if (!metodoEstatico && objeto.getClass()!=clase) {
					//TODO problemas en panel GPS Triump
//					throw new IllegalArgumentException("El objeto pasado no es de la clase "
//							+clase.getCanonicalName());
					return;
				}
				if(metodoEstatico)
					setText(String.format(formato, metodo.invoke(null, (Object[])null)));
				else
					setText(String.format(formato, metodo.invoke(objeto, (Object[])null)));
			} catch (IllegalArgumentException e) {
				// TODO Bloque catch generado automáticamente
				setEnabled(false);
				e.printStackTrace();
			} catch (IllegalAccessException e) {
				// TODO Bloque catch generado automáticamente
				setEnabled(false);
				e.printStackTrace();
			} catch (InvocationTargetException e) {
				// TODO Bloque catch generado automáticamente
				setEnabled(false);
				e.printStackTrace();
			}
		}
	}

	/** crea otro LabelDato con la mismas características que el actual */
	public LabelDato copiaParaResumen() {
		copiaResumen=new LabelDatoFormato(this);
		copiaResumen.copiaResumen=copiaResumen; //apunta a si mismo para indicar que esta en resumen
		copiaResumen.setEnabled(isEnabled());
		return copiaResumen;
	}
	
	public void borradaCopiaResumen() {
		copiaResumen=null;
	}
	
	public boolean tieneCopiaParaResumen() {
		return copiaResumen!=null;
	}
	
	public boolean esCopiaParaResumen() {
		return copiaResumen==this;
	}
}
