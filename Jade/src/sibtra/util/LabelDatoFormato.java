/**
 * 
 */
package sibtra.util;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

/**
 * Clase para todas aquellas etiquetas cuyo valor se actualiza invocando 
 * un método sin argumentos del objeto pasado
 * @author alberto
 *
 */
public class LabelDatoFormato extends LabelDato {

	/** metodo que invocar */
	Method 	metodo;
	/** formato a aplicar */
	String 	formato;
	/** clase del objeto sobre el que invocar el método */
	Class	clase;
	
	/**
	 * Crea etiqueta con texto inicial
	 * @param textoInicial
	 * @param clase clase cuyo método vamos a invocar
	 * @param nomMetodo nombre del metodo a invocar en cada actulización
	 * @param formato formato de {@link String.format} que se utilizara en cada actulización
	 */
	public LabelDatoFormato(String textoInicial,Class clase, String nomMetodo, String formato) {
		super(textoInicial);
		this.formato=formato;
		this.clase=clase;
		if(clase==null)
			throw new IllegalArgumentException("La clase pasada ha de ser != de null");
		try {
			this.metodo=clase.getMethod(nomMetodo,(Class[])null);
		} catch (SecurityException e) {
			throw new IllegalArgumentException("El metodo "+nomMetodo
					+ " no se puede invocar por razones de seguridad");
		} catch (NoSuchMethodException e) {
			throw new IllegalArgumentException("El metodo "+nomMetodo
					+ " no existe sin argumentos en la calse "
					+ clase.getCanonicalName());			
		}
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
				if (objeto.getClass()!=clase) {
					throw new IllegalArgumentException("El objeto pasado no es de la clase "
							+clase.getCanonicalName());
				}
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
}
