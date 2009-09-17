package sibtra.util;

import java.awt.Dimension;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;


/** 
 * Spinner de enteros que actualiza campo de un objeto a través de método get.
 * @author alberto
 *
 */
public class SpinnerInt extends JSpinner implements ChangeListener {
	
	private Object objeto;
	private Method metodo;
	private SpinnerNumberModel snm;

	public SpinnerInt(Object obj, String nomMetodo, int min, int max, int paso, int valIni) {
		construye(obj,nomMetodo,min,max,paso,valIni);
	}

	/** Constructor con valor inicial leido con método get correspondiente al set pasado */
	public SpinnerInt(Object obj, String nomMetodo, int min, int max, int paso) {
		if(obj==null)
			throw new IllegalArgumentException("El objeto pasado ha de ser != de null");
		String nomGet="";
		try {
			nomGet="get"+nomMetodo.substring(3);
			Method metGet=obj.getClass().getMethod(nomGet,(Class[])null);
			construye(obj,nomMetodo,min,max,paso,(Integer)(metGet.invoke(obj, (Object[])null)));
		} catch (SecurityException e) {
			throw new IllegalArgumentException("El metodo "+nomGet
					+ " no se puede invocar por razones de seguridad");
		} catch (NoSuchMethodException e) {
			throw new IllegalArgumentException("El metodo "+nomGet
					+ " no existe sin argumentos en la calse "
					+ obj.getClass().getCanonicalName());			
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

	/** Constructor con paso=1, y valor inicial leido con método get */
	public SpinnerInt(Object obj, String nomMetodo, int min, int max) {
		this(obj,nomMetodo,min,max,1);
	}

	private void construye(Object obj, String nomMetodo, int min, int max, int paso, int valIni) {
		this.objeto=obj;
		if(obj==null)
			throw new IllegalArgumentException("El objeto pasado ha de ser != de null");
		try {
			Class[] retorno={int.class};
			this.metodo=objeto.getClass().getMethod(nomMetodo,retorno);
		} catch (SecurityException e) {
			throw new IllegalArgumentException("El metodo "+nomMetodo
					+ " no se puede invocar por razones de seguridad");
		} catch (NoSuchMethodException e) {
			throw new IllegalArgumentException("El metodo "+nomMetodo
					+ " no existe con único argumento entero en clase "
					+ objeto.getClass().getCanonicalName());			
		}
		snm=new SpinnerNumberModel(valIni,min,max,paso);
		snm.addChangeListener(this);
		setModel(snm);
		setEnabled(true);
		//TODO determinar el tamaño de otra manera
		setMinimumSize(new Dimension(100,50));
		setPreferredSize(getMinimumSize());
	}

	public void stateChanged(ChangeEvent ce) {
		if(ce.getSource()==snm) {
			//invocamos método para fijar el nuevo valor
			try {
				Object[] argumento={new Integer(snm.getNumber().intValue())};
				metodo.invoke(objeto, argumento);
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
